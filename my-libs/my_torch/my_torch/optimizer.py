import numpy as np


class Optimizer:
    param_list: list

    def __init__(self, lr, weight_decay_rate=0.):
        self.lr = lr
        self.weight_decay_rate = weight_decay_rate

    def push_params(self, param_list):
        self.param_list = param_list

    def weight_decay(self):
        if self.weight_decay_rate <= 0:
            return

        for p in self.param_list:
            p.data -= self.lr * self.weight_decay_rate * p.data

    def step(self):
        pass


class SGD(Optimizer):
    def step(self):
        self.weight_decay()
        for p in self.param_list:
            p.data -= self.lr * p.grad


class Momentum(Optimizer):
    delta: list

    def __init__(self, lr, momentum=0.5, weight_decay_rate=0.0):
        super().__init__(lr, weight_decay_rate)
        self.momentum = momentum

    def push_params(self, param_list):
        super().push_params(param_list)
        length = len(param_list)

        self.delta = [None] * length
        for i in range(length):
            self.delta[i] = np.zeros(param_list[i].shape, dtype=np.float32)

    def step(self):
        self.weight_decay()

        for i in range(len(self.param_list)):
            p = self.param_list[i]

            self.delta[i] = self.momentum * self.delta[i] - self.lr * p.grad

            p.data += self.delta[i]


class AdaGrad(Optimizer):
    lr_decay: list

    def push_params(self, param_list):
        self.param_list = param_list
        length = len(param_list)

        self.lr_decay = [None] * length
        for i in range(length):
            self.lr_decay[i] = np.ones(param_list[i].shape, dtype=np.float32)

    def step(self):
        self.weight_decay()

        for i in range(len(self.param_list)):
            p = self.param_list[i]
            decay = self.lr_decay[i]

            decay += p.grad ** 2
            decay[decay > 1e4] = 1e4

            p.data -= self.lr * p.grad / decay ** 0.5
