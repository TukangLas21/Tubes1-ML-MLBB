from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @staticmethod
    @abstractmethod
    def loss(y_true, y_pred):
        pass

    @staticmethod
    @abstractmethod
    def derivative(y_true, y_pred):
        pass
    
class MSE(LossFunction):
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def derivative(y_true, y_pred):
        return (2 / y_true.shape[0]) * (y_pred - y_true)
    
class BinaryCrossEntropy(LossFunction):
    @staticmethod
    def loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # biar gak log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def derivative(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]
    
class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def loss(y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def derivative(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(1 / y_true.shape[0]) * (y_true / y_pred)