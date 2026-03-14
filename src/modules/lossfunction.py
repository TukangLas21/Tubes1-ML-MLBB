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