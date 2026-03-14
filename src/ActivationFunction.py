import os
import math
from abc import ABC, abstractmethod
import numpy as np

# activation function using strategy pattern

class ActivationFunction(ABC):
  def __init__(self):
    self.x = None 
    
  @abstractmethod
  def forward(self, x):
    pass
  
  @abstractmethod
  def backward(self, gradient_output):
    pass
       

class Linear(ActivationFunction):
  def forward(self, x):
    self.x = x
    
    return x
  
  def backward(self, gradient_output):
    return gradient_output * np.ones_like(self.x)


class Sigmoid(ActivationFunction):
  def forward(self, x):
    self.x = x
    
    return 1 / (1 + math.exp(-x))
  
  def backward(self, gradient_output):
    return gradient_output * self.forward(self.x) * (1 - self.forward(self.x))
  
  
class ReLU(ActivationFunction):
  def forward(self, x):
    self.x = x
    
    return max(0, x)
  
  def backward(self, gradient_output):
    return gradient_output * (self.x > 0)
  
  
class Tanh(ActivationFunction):
  def forward(self, x):
    self.x = x
    
    return math.tanh(x)
  
  def backward(self, gradient_output):
    return gradient_output * ((2 / math.exp(self.x) - math.exp(-self.x)) ** 2)
  
  
class Softmax(ActivationFunction):
  def forward(self, x):
    self.x = x
    
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    
    return [i / sum_exp_x for i in exp_x]
  
  def backward(self, gradient_output):
    pass # fck kok ribet bgt
  
  