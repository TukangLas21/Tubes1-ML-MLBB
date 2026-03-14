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
            initializer = self.initializer.initialize(shape=(input_dim, output_dim))
            layers.append(Layer(input_dim, output_dim, activation_fn, initializer))
        return layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X_train, y_train, X_val, y_val, verbose=1):
        pass