from abc import ABC, abstractmethod
import numpy as np

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
        return np.ones_like(z)
    
    
class Sigmoid(ActivationFunction):
    @staticmethod
    def activate(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z):
        forward_val = Sigmoid.activate(z)
        return forward_val * (1 - forward_val)
    
    
class ReLU(ActivationFunction):
    @staticmethod
    def activate(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        return (z > 0)
    
    
class Tanh(ActivationFunction):
    @staticmethod
    def activate(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z) ** 2    


class Softmax(ActivationFunction):
    @staticmethod
    def activate(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def derivative(z):
        softmax_output = Softmax.activate(z)
        n, C = softmax_output.shape
        
        jacobian_matrices = np.zeros((n, C, C))
        
        for i in range(n):
            softmax_output_i = softmax_output[i]
            jacobian_matrices[i] = np.diag(softmax_output_i) - np.outer(softmax_output_i, softmax_output_i)
        
        return jacobian_matrices