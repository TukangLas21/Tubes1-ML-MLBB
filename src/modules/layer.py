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

    # use_combined: softmax + cce
    def backward(self, d_a, use_combined=False):
        if use_combined:
            dz = d_a # TODO (verif): harusnya gini klo softmax dgn syarat lossnya CCE
        elif isinstance(self.activation_fn, Softmax):
            jacobian_matrices = type(self.activation_fn).derivative(self.z)
            dz = np.zeros_like(d_a)
            
            for i in range(d_a.shape[0]):
                dz[i] = np.dot(jacobian_matrices[i], d_a[i])
        else:
            dz = d_a * type(self.activation_fn).derivative(self.z)

        self.dW = np.dot(self.input.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)

        # pass gradient to previous layer
        return np.dot(dz, self.W.T)