import torch
from torch import nn
import utils


def vgg_block(num_conv, in_channels, out_channels):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    for _ in range(num_conv - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    return nn.Sequential(*layers, nn.BatchNorm2d(out_channels), nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU())


def vgg(in_channels, block_args):
    conv_layers = []
    for num_conv, out_channels in block_args:
        conv_layers.append(vgg_block(num_conv, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_layers, nn.Flatten(),
                     nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                     nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                     nn.Linear(4096, 10))


def vgg11():
    return vgg(1, [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)])


net = vgg(1, [(1, 16), (1, 32), (2, 64), (2, 128), (2, 128)])
loader, _ = utils.load_fashion_mnist(64, 224)
loss_func = torch.nn.CrossEntropyLoss()
manage = utils.ModelManager(net)
manage.train(loader, loss_func, 5)
