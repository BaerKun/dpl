import torch
from torch import nn
import utils


class ConvBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
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
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                         nn.BatchNorm2d(out_channels), nn.ReLU(),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def densenet(in_channels, growth_rate, num_convs: list[int]):
    blocks = [nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
              nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

    in_channels = 64
    for i, num_conv in enumerate(num_convs):
        blocks.append(dense_block(num_conv, in_channels, growth_rate))
        in_channels += growth_rate * num_conv
        if i + 1 != len(num_convs):
            blocks.append(transition_block(in_channels, in_channels // 2))
            in_channels //= 2

    return nn.Sequential(*blocks, nn.BatchNorm2d(in_channels), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, 10))


net = densenet(1, 32, [4, 4, 4, 4])
mm = utils.ModelManager('../../model/densenet.pt')

# loader = utils.load_fashion_mnist(128, 96)
# mm.train(loader, nn.CrossEntropyLoss(), 10, 0.00001)
# mm.save('../../model/densenet.pt')

loader, labels = utils.load_fashion_mnist(128, 96, False, get_labels=True)
mm.test(loader, utils.score_acc)

for x, y in loader:
    if not mm.predict_with_image(x, labels):
        break
