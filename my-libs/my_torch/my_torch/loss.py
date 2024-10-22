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

        loss = np.mean(self.input_target ** 2)
        return loss

    def backward(self):
        grad = self.input_target * 2 / self.input_target.shape[-1]
        return grad


class CrossEntropyLoss(LossFunction):
    def __call__(self, tensor, target):
        self.target = target
        self.in_tensor = tensor + 1e-7
        self.one_minus_tensor = 1.0000001 - tensor

        loss = np.sum(target * np.log(self.one_minus_tensor / self.in_tensor)
                      - np.log(self.one_minus_tensor), axis=0).mean()
        return loss

    def backward(self):
        grad = (self.in_tensor - self.target) / (self.one_minus_tensor * self.in_tensor)
        return grad
