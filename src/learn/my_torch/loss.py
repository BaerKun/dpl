import numpy as np


class LossFunction:
    param_list = None
    reg_rate: float

    def __call__(self, tensor, target):
        pass

    def backward(self):
        pass


class MSELoss(LossFunction):
    def __call__(self, tensor, target):
        self.input_target = tensor - target

        loss = np.mean(self.input_target ** 2) / 2
        return loss

    def backward(self):
        grad = self.input_target * 2 / self.input_target.shape[-1]
        return grad


class CrossEntropyLoss(LossFunction):
    target: np.ndarray = None
    in_tensor: np.ndarray = None
    one_minus_tensor: np.ndarray = None

    def __call__(self, tensor, target):
        if self.target is None:
            self.target = np.zeros_like(target)
            self.in_tensor = np.zeros_like(tensor)
            self.one_minus_tensor = np.zeros_like(tensor)

        self.target[:] = target
        np.add(tensor, 1e-7, out=self.in_tensor)
        np.subtract(1.0000001, tensor, out=self.one_minus_tensor)
        loss = np.sum(target * np.log(self.one_minus_tensor / self.in_tensor)
                      - np.log(self.one_minus_tensor), axis=1)

        return np.mean(loss)

    def backward(self):
        grad = (self.in_tensor - self.target) / (self.one_minus_tensor * self.in_tensor)
        return grad
