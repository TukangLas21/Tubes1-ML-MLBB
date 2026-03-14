import math
from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
  @abstractmethod
  def forward(self, y_pred, y_true):
    pass
  
  @abstractmethod
  def backward(self, y_pred, y_true):
    pass
  

class MSE(LossFunction):
  def forward(self, y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)
  
  def backward(self, y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)
  

class BinaryCrossEntropy(LossFunction):
  def forward(self, y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
  
  def backward(self, y_pred, y_true):
    pass # TODO
  
  
class CategoricalCrossEntropy(LossFunction):
  def forward(self, y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
  
  def backward(self, y_pred, y_true):
    pass # TODO