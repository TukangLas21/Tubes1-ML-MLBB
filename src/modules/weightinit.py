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
    

# std = sqrt(2 / (nin + out))
class XavierNormalInitializer(Initializer):
    def __init__(self, mean=0.0, seed: int | None = 42):
        super().__init__(seed)
        self.mean = mean
        
    def initialize(self, shape):
        n_in = shape[0]
        n_out = shape[1]
        
        std = np.sqrt(2.0 / (n_in + n_out))
        
        return np.random.normal(loc=self.mean, scale=std, size=shape)
        

# x = sqrt(6 / (nin + nout)), uniform [-x, x]
class XavierUniformInitializer(Initializer):
    def __init__(self, seed: int | None = 42):
        super().__init__(seed)

    def initialize(self, shape):
        n_in = shape[0]
        n_out = shape[1]
        
        limit = np.sqrt(6.0 / (n_in + n_out))
        
        return np.random.uniform(-limit, limit, size=shape)
    

# std = sqrt(2 / nin)
class HeNormalInitializer(Initializer):
    def __init__(self, mean=0.0, seed: int | None = 42):
        super().__init__(seed)
        self.mean = mean
        
    def initialize(self, shape):
        n_in = shape[0]
        
        std = np.sqrt(2.0 / n_in)
        
        return np.random.normal(loc=self.mean, scale=std, size=shape)
    

# x = sqrt(6 / nin)
class HeUniformInitializer(Initializer):
    def __init__(self, seed: int | None = 42):
        super().__init__(seed)

    def initialize(self, shape):
        n_in = shape[0]
        
        limit = np.sqrt(6.0 / n_in)
        
        return np.random.uniform(-limit, limit, size=shape)