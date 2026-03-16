from __future__ import annotations

import numpy as np
from tqdm import tqdm
from typing import Literal, overload

from modules.autograd import Tensor
from modules.config import ModelConfig
from modules.layer import Layer
from modules.weightinit import Initializer
from modules.rmsnorm import RMSNorm

class FFNN:
    def __init__(self, config: ModelConfig, initializer: Initializer):
        self.config = config
        self.initializer = initializer
        self.layers = self._initialize_layers()
        self.history = {'train_loss': [], 'val_loss': []}

    def _initialize_layers(self) -> list[Layer]:
        layers = []
        for i in range(len(self.config.layers_dims) - 1):
            input_dim = self.config.layers_dims[i]
            output_dim = self.config.layers_dims[i + 1]
            activation_fn = self.config.activation_fn[i]
            W = self.initializer.initialize((input_dim, output_dim))
            if self.config.rmsnorm and i != len(self.config.layers_dims) - 2: # except output layer
                rmsnorm = RMSNorm(output_dim)
                layers.append(Layer(input_dim, output_dim, activation_fn, W, rmsnorm))
            else:
                layers.append(Layer(input_dim, output_dim, activation_fn, W))
        return layers
    
    @overload
    def forward(self, x: np.ndarray, return_tensor: Literal[True]) -> Tensor:
        ...

    @overload
    def forward(self, x: np.ndarray, return_tensor: Literal[False] = False) -> np.ndarray:
        ...

    def forward(self, x: np.ndarray, return_tensor: bool = False) -> Tensor | np.ndarray:
        x_t = Tensor(x, requires_grad=False)
        for layer in self.layers:
            x_t = layer.forward(x_t)

        if return_tensor:
            return x_t
        return x_t.numpy()

    def backward(self, y_true: Tensor, y_pred: Tensor) -> None:
        loss_t = self.config.loss_fn.loss(y_true, y_pred)
        loss_t.backward()
        grad = None
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self) -> None:
        lr = self.config.learning_rate
        reg = self.config.regularization
        lam = self.config.reg_lambda

        for layer in self.layers:
            if reg == 'l1':
                reg_grad = lam * np.sign(layer.W.data)
            elif reg == 'l2':
                reg_grad = lam * layer.W.data
            else:
                reg_grad = 0.0

            w_grad = layer.dW if layer.dW is not None else 0.0
            b_grad = layer.db if layer.db is not None else 0.0

            layer.W.data -= lr * (w_grad + reg_grad)
            layer.b.data -= lr * b_grad
            
            if layer.rmsnorm is not None and layer.rmsnorm.gamma.grad is not None:
                layer.rmsnorm.gamma.data -= lr * layer.rmsnorm.gamma.grad

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        np.random.seed(self.config.random_state)
        self.history = {'train_loss': [], 'val_loss': []}

        n = X_train.shape[0]
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        verbose = self.config.verbose

        progress_bar = tqdm(range(epochs), desc='Training') if verbose == 1 else None
        epoch_range = progress_bar if progress_bar is not None else range(epochs)

        for epoch in epoch_range:
            # shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]

            batch_losses = []
            for start in range(0, n, batch_size):
                X_b = X_s[start:start + batch_size]
                y_b = y_s[start:start + batch_size]

                # zero gradients from prev batch
                for layer in self.layers:
                    layer.W.zero_grad()
                    layer.b.zero_grad()
                    if layer.rmsnorm is not None:
                        layer.rmsnorm.gamma.zero_grad()

                    layer.dW = None
                    layer.db = None

                y_b_t = Tensor(y_b, requires_grad=False)
                y_pred_t = self.forward(X_b, return_tensor=True)
                loss_t = self.config.loss_fn.loss(y_b_t, y_pred_t)
                self.backward(y_b_t, y_pred_t)

                batch_losses.append(float(loss_t.data))
                self._update_weights()

            train_loss = float(np.mean(batch_losses))
            self.history['train_loss'].append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_t = Tensor(y_val, requires_grad=False)
                y_val_pred = self.forward(X_val, return_tensor=True)
                val_loss = float(self.config.loss_fn.loss(y_val_t, y_val_pred).data)
                self.history['val_loss'].append(val_loss)

            if progress_bar is not None:
                postfix = {'train_loss': f'{train_loss:.4f}'}
                if val_loss is not None:
                    postfix['val_loss'] = f'{val_loss:.4f}'
                progress_bar.set_postfix(postfix)

        return self.history
