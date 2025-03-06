import torch
from torch import nn
import utils


class Inception(nn.Module):
    def __init__(self, in_channels, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int):
        super(Inception, self).__init__()
        self.p1 = nn.Conv2d(in_channels, c1, kernel_size=1, bias=False)
        self.p2 = nn.Sequential(nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
                                nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1, bias=False))
        self.p3 = nn.Sequential(nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
                                nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2, bias=False))
        self.p4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.Conv2d(in_channels, c4, kernel_size=1, bias=False))
        self.norm = nn.BatchNorm2d(c1 + c2[1] + c3[1] + c4)

    def forward(self, x):
        return torch.relu(self.norm(torch.cat((self.p1(x), self.p2(x), self.p3(x), self.p4(x)), dim=1)))


num_classes = 10
net = nn.Sequential(
    # b1
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(),
    # b2
    nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(192),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.ReLU(),
    # b3
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    # b4
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    # b5
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    # b6
    nn.Linear(1024, num_classes)
)

mm = utils.ModelManager(net, "../../weights/google_net.pt")
loader = utils.load_fashion_mnist(128, 96)
loss_f = nn.CrossEntropyLoss()
mm.train(loader, loss_f, 5)
mm.test(loader, utils.score_acc)

loader = utils.load_fashion_mnist(128, 224, False)
mm.test(loader, utils.score_acc)

mm.save("../../weights/google_net.pt")
