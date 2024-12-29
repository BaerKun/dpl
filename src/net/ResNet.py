import torch
from torch import nn
import utils


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_channels))
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, x):
        y = self.conv3x3(x)
        if self.conv1x1 is not None:
            y += self.conv1x1(x)
        return torch.relu(y)


def resnet_block(in_channels, out_channels, num_residuals, use_1x1conv=True):
    blocks = []
    for _ in range(num_residuals):
        if use_1x1conv:
            blocks.append(Residual(in_channels, out_channels, True, 2))
            use_1x1conv = False
        else:
            blocks.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blocks)

net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    resnet_block(64, 64, 2, False),
    resnet_block(64, 128, 2),
    resnet_block(128, 256, 2),
    resnet_block(256, 512, 2),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

mm = utils.ModelManager("../../model/resnet.pt")
loader = utils.load_fashion_mnist(128, 96)
mm.train(loader, nn.CrossEntropyLoss(), 10, 0.000001)
loader = utils.load_fashion_mnist(128, 96, False)
mm.test(loader, mode="acc")

# mm.save("../../model/resnet.pt")