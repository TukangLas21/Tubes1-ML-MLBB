import numpy as np

class Weights:
  @staticmethod
  def zeros(shape):
    return np.zeros(shape)
  
  @staticmethod
  def random_uniform(shape, low=-1.0, high=1.0, random_state=None):
    if random_state is not None:
      np.random.seed(random_state)
      
    return np.random.uniform(low, high, size=shape)
  
  @staticmethod
  def random_normal(shape, mean=0.0, std=1.0, random_state=None):
    if random_state is not None:
      np.random.seed(random_state)
      
    return np.random.normal(mean, std, size=shape) 
    