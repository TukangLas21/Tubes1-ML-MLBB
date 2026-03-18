import numpy as np

from modules.activationfunction import ActivationFunction
from modules.autograd import Tensor
from modules.rmsnorm import RMSNorm


class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation_fn: ActivationFunction, weights: np.ndarray, rmsnorm: RMSNorm=None):
        self.rmsnorm = rmsnorm
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        self.W = Tensor(weights, requires_grad=True)
        self.b = Tensor(np.zeros((1, output_dim)), requires_grad=True)

        self.input: Tensor | None = None
        self.z: Tensor | None = None
        self.a: Tensor | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, input_data: Tensor) -> Tensor:
        self.input = input_data
        self.z = self.input @ self.W + self.b
        
        if self.rmsnorm is not None:
            self.z = self.rmsnorm.forward(self.z)
            
        self.a = self.activation_fn.activate(self.z)
        return self.a

    def backward(self, grad: np.ndarray | None = None) -> np.ndarray | None:
        self.dW = self.W.grad
        self.db = self.b.grad

        if self.input is None:
            raise ValueError("Layer input is not available.")

        return self.input.grad

