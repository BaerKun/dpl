import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels + growth_rate)

    def forward(self, x):
        y = self.conv(x)
        y = torch.cat((x, y), dim=1)
        return torch.relu(self.bn(y))


def dense_block(num_conv, in_channels, growth_rate):
    blocks = []
    for i in range(num_conv):
        blocks.append(ConvBlock(in_channels + i * growth_rate, growth_rate))
    return nn.Sequential(*blocks)


def transition_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(),
                         nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate, num_classes, num_convs: list[int]):
        super().__init__()

        head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        blocks = []
        in_channels = 64
        for i, num_conv in enumerate(num_convs):
            blocks.append(dense_block(num_conv, in_channels, growth_rate))
            in_channels += growth_rate * num_conv

            if i + 1 != len(num_convs):
                blocks.append(transition_block(in_channels, in_channels // 2))
                in_channels //= 2

        self.layers =  nn.Sequential(
            head,
            *blocks,
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(in_channels, num_classes))

    def forward(self, x):
        return self.layers(x)