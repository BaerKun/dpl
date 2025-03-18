from torch import nn


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU())


class NiN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten())

    def forward(self, x):
        return self.layers(x)
