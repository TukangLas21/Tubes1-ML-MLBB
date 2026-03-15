from typing import List

from modules.activationfunction import ActivationFunction
from modules.lossfunction import LossFunction

class ModelConfig:
    def __init__(self, layers_dims, activation_fn: List[ActivationFunction], loss_fn: LossFunction, 
                 learning_rate=0.01, epochs=100, batch_size=32, verbose=1,
                 regularization="l1", reg_lambda=0.01, random_state=42):
        self.layers_dims = layers_dims
        self.activation_fn: List[ActivationFunction] = activation_fn
        self.loss_fn: LossFunction = loss_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.random_state = random_state