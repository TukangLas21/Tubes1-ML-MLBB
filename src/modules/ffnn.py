import numpy as np
from tqdm import tqdm

from modules.config import ModelConfig
from modules.layer import Layer
from modules.weightinit import Initializer

class FFNN:
    def __init__(self, config: ModelConfig, initializer: Initializer):
        self.config = config
        self.initializer = initializer
        self.layers = self._initialize_layers()
        self.history = {'train_loss': [], 'val_loss': []}

    def _initialize_layers(self):
        layers = []
        for i in range(len(self.config.layers_dims) - 1):
            input_dim = self.config.layers_dims[i]
            output_dim = self.config.layers_dims[i + 1]
            activation_fn = self.config.activation_fn[i]
            W = self.initializer.initialize((input_dim, output_dim))
            layers.append(Layer(input_dim, output_dim, activation_fn, W))
        return layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self):
        lr  = self.config.learning_rate
        reg = self.config.regularization
        lam = self.config.reg_lambda

        for layer in self.layers:
            if reg == 'l1':
                reg_grad = lam * np.sign(layer.W)
            elif reg == 'l2':
                reg_grad = lam * layer.W
            else:
                reg_grad = 0.0

            layer.W -= lr * (layer.dW + reg_grad)
            layer.b -= lr * layer.db


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        np.random.seed(self.config.random_state)
        self.history = {'train_loss': [], 'val_loss': []}

        n          = X_train.shape[0]
        batch_size = self.config.batch_size
        epochs     = self.config.epochs
        verbose    = self.config.verbose

        epoch_range = tqdm(range(epochs), desc='Training') if verbose == 1 else range(epochs)

        for epoch in epoch_range:
            # Shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]

            batch_losses = []
            for start in range(0, n, batch_size):
                X_b = X_s[start:start + batch_size]
                y_b = y_s[start:start + batch_size]

                y_pred = self.forward(X_b)
                batch_losses.append(self.config.loss_fn.loss(y_b, y_pred))

                loss_grad = self.config.loss_fn.derivative(y_b, y_pred)
                self.backward(loss_grad)
                self._update_weights()

            train_loss = float(np.mean(batch_losses))
            self.history['train_loss'].append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = float(self.config.loss_fn.loss(y_val, y_val_pred))
                self.history['val_loss'].append(val_loss)

            if verbose == 1:
                postfix = {'train_loss': f'{train_loss:.4f}'}
                if val_loss is not None:
                    postfix['val_loss'] = f'{val_loss:.4f}'
                epoch_range.set_postfix(postfix)

        return self.history
