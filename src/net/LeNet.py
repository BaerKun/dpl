import torch
import torchvision
import utils

net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 6, 5, 1, 2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(6, 16, 5, 1, 0),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(400, 120),
    torch.nn.ReLU(),
    torch.nn.Linear(120, 84),
    torch.nn.ReLU(),
    torch.nn.Linear(84, 10)
)

loader, _ = utils.load_fashion_mnist(4096)
loss_func = torch.nn.CrossEntropyLoss()

model_manager = utils.ModelManager(net)
model_manager.train(loader, loss_func, 10, 0.1)