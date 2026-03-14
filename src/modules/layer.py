import numpy as np

from modules.activationfunction import ActivationFunction, Softmax


class Layer:
    def __init__(self, input_dim, output_dim, activation_fn: ActivationFunction, weights: np.ndarray):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        self.W = weights
        self.b = np.zeros((1, output_dim))

        self.input = None
        self.z = None
        self.a = None

        self.dW = None
        self.db = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.W) + self.b
        self.a = self.activation_fn.activate(self.z)
        return self.a

    def backward(self, d_a):
        if isinstance(self.activation_fn, Softmax):
            dz = 0 # TODO: Implement softmax backward (requires special handling)
        else:
            dz = d_a * type(self.activation_fn).derivative(self.z)

        self.dW = np.dot(self.input.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)

        # Pass gradient to previous layer
        return np.dot(dz, self.W.T)