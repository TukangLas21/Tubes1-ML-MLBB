from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def initialize(self, shape, **kwargs):
        pass

class ZerosInitializer(Initializer):
    def initialize(self, shape, **kwargs):
        return np.zeros(shape)

class RandomUniformInitializer(Initializer):
    def initialize(self, shape, **kwargs):
        low = kwargs.get('low', -0.5)
        high = kwargs.get('high', 0.5)
        return np.random.uniform(low, high, size=shape)

class RandomNormalInitializer(Initializer):
    def initialize(self, shape, **kwargs):
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 0.05)
        return np.random.normal(loc=mean, scale=std, size=shape)