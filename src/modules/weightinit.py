from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    def __init__(self, seed=42):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def initialize(self, shape):
        pass


class ZerosInitializer(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)


class RandomUniformInitializer(Initializer):
    def __init__(self, low=-0.5, high=0.5, seed=42):
        super().__init__(seed)
        self.low = low
        self.high = high

    def initialize(self, shape):
        return np.random.uniform(self.low, self.high, size=shape)


class RandomNormalInitializer(Initializer):
    def __init__(self, mean=0.0, variance=0.01, seed=42):
        super().__init__(seed)
        self.mean = mean
        self.std = np.sqrt(variance)

    def initialize(self, shape):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)