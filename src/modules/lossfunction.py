from abc import ABC, abstractmethod

import numpy as np

from modules.autograd import Tensor

class LossFunction(ABC):
    @staticmethod
    @abstractmethod
    def loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
class MSE(LossFunction):
    @staticmethod
    def loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return ((y_true - y_pred) * (y_true - y_pred)).mean()
    
    @staticmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (2 / y_true.shape[0]) * (y_pred - y_true)
    
class BinaryCrossEntropy(LossFunction):
    @staticmethod
    def loss(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-15) -> Tensor:
        y_p = y_pred * Tensor(1 - 2 * epsilon) + Tensor(epsilon)
        return -((y_true * y_p.log()) + ((Tensor(1) - y_true) * (Tensor(1) - y_p).log())).mean()
    
    @staticmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (1 / y_true.shape[0]) * (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)
    
class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def loss(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-15) -> Tensor:
        y_p = y_pred * Tensor(1 - 2 * epsilon) + Tensor(epsilon)
        return -(y_true * y_p.log()).sum(axis=1).mean()
    
    @staticmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(1 / y_true.shape[0]) * (y_true / y_pred)
