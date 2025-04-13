import numpy as np


class Param:
    def __init__(self, var):
        self.data = var
        self.grad = np.ndarray(self.data.shape, dtype=np.float32)
        self.shape = self.grad.shape


def xavier(shape, n) -> Param:
    data = np.random.normal(0.0, np.sqrt(2.0 / n), shape).astype(np.float32)
    return Param(data)
