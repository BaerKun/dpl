from torch import nn
import utils

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels), nn.ReLU())

net = nn.Sequential(nin_block(1, 96, kernel_size=11, strides=4, padding=0),
                    nn.MaxPool2d(3, stride=2),
                    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
                    nn.MaxPool2d(3, stride=2),
                    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
                    nn.MaxPool2d(3, stride=2),
                    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten())

loader = utils.load_fashion_mnist(96, 224)
loss_f = nn.CrossEntropyLoss()
mm = utils.ModelManager(net)
mm.train(loader, loss_f, 5)
