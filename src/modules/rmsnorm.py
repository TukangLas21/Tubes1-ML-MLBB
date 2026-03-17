import numpy as np
from modules.autograd import Tensor

class RMSNorm:
  # gamma = 1
  def __init__(self, dimension: int, epsilon: float = 1e-7):
    self.dimension = dimension
    self.epsilon = epsilon
    self.gamma = Tensor(np.ones((1, dimension)), requires_grad=True)
    
  def forward(self, z: Tensor) -> Tensor:
    return z.rmsnorm(self.epsilon) * self.gamma
  
  def zero_grad(self) -> None:
    self.gamma.zero_grad()
  
  
    