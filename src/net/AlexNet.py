import torch, torchvision
from torch import nn
import utils

AlexNet = torch.nn.Sequential(
    nn.Conv2d(1, 96, 11, 4, 1),
    nn.MaxPool2d(3, 2), nn.ReLU(),
    nn.Conv2d(96, 256, 5, 1, 2), nn.BatchNorm2d(256),
    nn.MaxPool2d(3, 2), nn.ReLU(),
    nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
    nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
    nn.Conv2d(384, 256, 3, 1, 1), nn.BatchNorm2d(256),
    nn.MaxPool2d(3, 2), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 10)
)

loader = utils.load_fashion_mnist(96, 224)
loss_func = torch.nn.CrossEntropyLoss()

model_manager = utils.ModelManager(AlexNet)
model_manager.train(loader, loss_func, 5)
