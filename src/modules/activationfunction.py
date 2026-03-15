from abc import ABC, abstractmethod
import numpy as np

from modules.autograd import Tensor

class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def activate(z):
        pass

    @staticmethod
    @abstractmethod
    def derivative(z):
        pass


class Linear(ActivationFunction):
    @staticmethod
    def activate(z):
        return z

    @staticmethod
    def derivative(z):
        return Tensor(np.ones_like(z.data))
    
    
class Sigmoid(ActivationFunction):
    @staticmethod
    def activate(z):
        return z.sigmoid()
    
    @staticmethod
    def derivative(z):
        forward_val = Sigmoid.activate(z)
        return forward_val * (Tensor(1) - forward_val)
    
    
class ReLU(ActivationFunction):
    @staticmethod
    def activate(z):
        return z.relu()

    @staticmethod
    def derivative(z):
        return (z.data > 0).astype(float)
    
    
class Tanh(ActivationFunction):
    @staticmethod
    def activate(z):
        return z.tanh()

    @staticmethod
    def derivative(z):
        return Tensor(1) - z.tanh() ** Tensor(2)


class Softmax(ActivationFunction):
    @staticmethod
    def activate(z):
        return z.softmax()

    @staticmethod
    def derivative(z):
        softmax_output = Softmax.activate(z)
        if softmax_output.ndim == 1:
            softmax_output = softmax_output.reshape(1, -1) 
        
        n, C = softmax_output.shape
        
        jacobian_matrices = np.zeros((n, C, C))
        
        for i in range(n):
            softmax_output_i = softmax_output[i]
            jacobian_matrices[i] = np.diag(softmax_output_i) - np.outer(softmax_output_i, softmax_output_i)
        
        return jacobian_matrices