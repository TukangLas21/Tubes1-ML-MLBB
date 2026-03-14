import numpy as np

from modules.activationfunction import ActivationFunction
from modules.weightinit import Initializer

class Layer:
    def __init__(self, input_dim, output_dim, activation_fn: ActivationFunction, initializer: Initializer):
        self.W = initializer.initialize((input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        
        self.activation_fn = activation_fn
        
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

    def backward(self, dz):
        self.dW = np.dot(self.input.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        
        # Gradient to pass to the previous layer
        return np.dot(dz, self.W.T)