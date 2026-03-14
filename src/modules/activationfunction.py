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

# contoh
class Linear(ActivationFunction):
    @staticmethod
    def activate(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones_like(z)