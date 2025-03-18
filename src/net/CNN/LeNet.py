import torch


class LeNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, 1, 2),
            torch.nn.MaxPool2d(2, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(6, 16, 5, 1, 0),
            torch.nn.MaxPool2d(2, 2), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(400, 120), torch.nn.ReLU(),
            torch.nn.Linear(120, 84), torch.nn.ReLU(),
            torch.nn.Linear(84, num_classes))

    def forward(self, x):
        return self.layers(x)
