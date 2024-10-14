from typing import Any

import numpy as np
from numpy import floating

'''
所有支持 np.ndarray 的函数都可传递 float
'''


def derivative(_function, _x: np.ndarray) -> np.ndarray:
    _delta = 1e-4
    return (_function(_x + _delta) - _function(_x - _delta)) * 5e3


def gradient(_function, _x: np.ndarray) -> np.ndarray:
    _delta = 1e-4
    _reciprocal = 5e3
    _grad = np.zeros_like(_x)

    for i in range(_x.size):
        _tmp = _x[i]
        _x[i] += _delta
        _front = _function(_x)

        _x[i] = _tmp - _delta
        _rear = _function(_x)
        _grad[i] = (_front - _rear) * _reciprocal

        _x[i] = _tmp

    return _grad


def gradient_descent(_function, _x: np.ndarray, lr: float = 0.01, step_num: int = 100) -> np.ndarray:
    _current = _x.copy()

    for i in range(step_num):
        _grad = gradient(_function, _current)
        _current -= _grad * lr

    return _current


def mean_squared_error(_x: np.ndarray, _target: np.ndarray):
    return np.mean((_x - _target) ** 2)


def cross_entropy_error(_x: np.ndarray, _target: np.ndarray) -> float:
    return -np.sum(_target * np.log(_x + 1e-7))


def sigmoid(_x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-_x))


def softmax(_x: np.ndarray) -> np.ndarray:
    _max = np.max(_x)
    _exp = np.exp(_x - _max)
    _sum_reciprocal = 1 / np.sum(_exp)

    return _exp * _sum_reciprocal


def func_1(_x):
    return sigmoid(_x)


def func_2(_x):
    return sigmoid(_x[0] + _x[1])


if __name__ == '__main__':
    x = np.random.rand(10)
    y = np.random.rand(2)
    print(x, '\n', y)
