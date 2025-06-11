import torch


def acc(y_hat: torch.Tensor, y: torch.Tensor, argmax=False):
    if argmax:
        y_hat = y_hat.argmax(dim=-1)
    return (y_hat == y).mean(dtype=torch.float32)


def mae(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.abs(y_hat - y).mean()


def mse(y_hat: torch.Tensor, y: torch.Tensor):
    return ((y_hat - y) ** 2).mean()


def r2(y_hat: torch.Tensor, y: torch.Tensor):
    return 1. - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
