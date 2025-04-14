from torch import nn


def vgg_block(num_conv, in_channels, out_channels):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    for _ in range(num_conv - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    return nn.Sequential(*layers, nn.BatchNorm2d(out_channels), nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU())


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes, block_args: list[(int, int)]):
        super().__init__()
        conv_layers = []
        for num_conv, out_channels in block_args:
            conv_layers.append(vgg_block(num_conv, in_channels, out_channels))
            in_channels = out_channels

        self.layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.layers(x)
