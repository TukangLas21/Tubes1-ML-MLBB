import numpy as np
from ActivationFunction import ActivationFunction

class Neuron:
  def __init__(self, weights=None, bias=None, inputs=None, learning_rate=None, activation_function: ActivationFunction=None):
    self.weights = weights
    self.bias = bias
    self.gradient_weights = None
    self.gradient_bias = None
    self.inputs = inputs
    self.activation_function = activation_function
    self.learning_rate = learning_rate
    
  def compute_net_input(self):
    return np.dot(self.weights, self.inputs) + self.bias
  
  def forward(self):
    return self.activation_function.forward()
  
  def backward(self, output_gradient):
    activation_gradient = self.activation_function.backward(output_gradient)
    self.gradient_weights = activation_gradient * self.inputs
    self.gradient_bias = activation_gradient
    return output_gradient * activation_gradient * self.weights
  
  # ngikut tutorial razi
  def update_parameters(self):
    self.weights -= self.learning_rate * self.gradient_weights
    self.bias -= self.learning_rate * self.gradient_bias
  