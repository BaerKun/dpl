import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Conv2d(1, 96, 11, 4, 1),
            nn.MaxPool2d(3, 2), nn.ReLU(),
            nn.Conv2d(96, 256, 5, 1, 2), nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2), nn.ReLU(),
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.layers(x)