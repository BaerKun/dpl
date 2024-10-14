import numpy as np


class Optimizer:
    param_list: list
    grad_k: np.float32

    def __init__(self, lr, weight_decay_rate=0.):
        self.lr = lr
        self.weight_decay_rate = weight_decay_rate

    def push_params(self, param_list):
        self.param_list = param_list

    def init_grad_k(self):
        self.grad_k = self.lr
        if self.weight_decay_rate > 0.0:
            sum_l2 = 0.0

            for param in self.param_list:
                sum_l2 += np.linalg.norm(param.grad)

            self.grad_k *= 1.0 + self.weight_decay_rate * sum_l2

    def step(self):
        pass


class SGD(Optimizer):
    def step(self):
        self.init_grad_k()
        for p in self.param_list:
            p.data -= self.grad_k * p.grad


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
        self.init_grad_k()

        for i in range(len(self.param_list)):
            p = self.param_list[i]

            self.delta[i] = self.momentum * self.delta[i] - self.grad_k * p.grad

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
        self.init_grad_k()

        for i in range(len(self.param_list)):
            p = self.param_list[i]
            decay = self.lr_decay[i]

            decay += p.grad ** 2
            decay[decay > 1e8] = 1e8

            p.data -= self.grad_k * p.grad / decay ** 0.5
