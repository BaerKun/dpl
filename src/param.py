import numpy as np


class Param:
    def __init__(self, var):
        self.data = var
        self.grad = np.ndarray(self.data.shape, dtype=np.float32)
        self.shape = self.grad.shape


def xavier(*shape: int) -> Param:
    n = shape[0]
    return Param(np.random.randn(*shape).astype(np.float32) / n ** 0.5)


def he(*shape: int) -> Param:
    n = shape[0]
    return Param(np.random.randn(*shape).astype(np.float32) * (2.0 / n) ** 0.5)
