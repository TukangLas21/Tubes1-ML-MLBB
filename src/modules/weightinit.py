from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    def __init__(self, seed: int | None = 42):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError


class ZerosInitializer(Initializer):
    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape)


class RandomUniformInitializer(Initializer):
    def __init__(self, low: float = -0.5, high: float = 0.5, seed: int | None = 42):
        super().__init__(seed)
        self.low = low
        self.high = high

    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=shape)


class RandomNormalInitializer(Initializer):
    def __init__(self, mean: float = 0.0, variance: float = 0.01, seed: int | None = 42):
        super().__init__(seed)
        self.mean = mean
        self.std = np.sqrt(variance)

    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)
