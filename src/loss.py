import numpy as np


class Loss(np.ndarray):
    def __new__(cls, value, backward_callback):
        obj = np.array(value, dtype=np.float32).view(cls)
        obj._backward = backward_callback

        return obj

    def backward(self):
        return self._backward()


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

        loss_value = np.mean(self.input_target ** 2)
        return Loss(loss_value, self._backward)

    def _backward(self):
        grad = self.input_target * 2 / self.input_target.shape[-1]
        return grad


class CrossEntropyLoss(LossFunction):
    def __call__(self, tensor, target):
        self.target = target
        self.in_tensor = tensor + 1e-7
        self.one_minus_tensor = 1.0000001 - tensor

        loss_value = np.sum(target * np.log(self.one_minus_tensor / self.in_tensor)
                            - np.log(self.one_minus_tensor), axis=0).mean()
        return Loss(loss_value, self._backward)

    def _backward(self):
        grad = (self.in_tensor - self.target) / (self.one_minus_tensor * self.in_tensor)
        return grad
