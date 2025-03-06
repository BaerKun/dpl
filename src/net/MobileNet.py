import torch
from torch import nn
import utils


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层(fully connection)，注意力评分函数
        # 这里使用的是加性注意力，但因为是自注意力，且查询和键相同，所以合并了 Wq 和 Wk
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),

            # 原地操作，不新开内存
            # 这不会影响反向传播的正确性，只要正向传播不再需要输入值
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        b, c = x.shape[:2]
        q = self.avg_pool(x).view(b, c)
        alpha = self.fc(q).view(b, c, 1, 1)
        return x * alpha


class Bneck(nn.Module):
    def __init__(self, in_channels, expand_size, out_channels, kernel_size, stride, use_se, activation):
        super().__init__()

        self.residual = in_channels == out_channels and stride == 1

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            activation
        ) if in_channels != expand_size else None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            activation
        )

        self.se = SEAttention(expand_size) if use_se else None

        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_size, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        y = x
        if self.expand_conv is not None:
            y = self.expand_conv(x)

        y = self.depth_conv(y)
        if self.se is not None:
            y = self.se(y)

        y = self.project_conv(y)
        if self.residual:
            y += x
        return y


class MobileNetV3(nn.Module):
    def __init__(self, in_channels, num_classes, use_large=False):
        super().__init__()

        relu = nn.ReLU(inplace=True)
        h_swish = nn.Hardswish(inplace=True)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), h_swish
        )

        if use_large:
            self.blocks = nn.Sequential(
                Bneck(16, 16, 16, 3, 1, False, relu),
                Bneck(16, 64, 24, 3, 2, False, relu),
                Bneck(24, 72, 24, 3, 1, False, relu),
                Bneck(24, 72, 40, 5, 2, True, relu),
                Bneck(40, 120, 40, 5, 1, True, relu),
                Bneck(40, 120, 40, 5, 1, True, relu),
                Bneck(40, 240, 80, 3, 2, False, h_swish),
                Bneck(80, 200, 80, 3, 1, False, h_swish),
                Bneck(80, 184, 80, 3, 1, False, h_swish),
                Bneck(80, 184, 80, 3, 1, False, h_swish),
                Bneck(80, 480, 112, 3, 1, True, h_swish),
                Bneck(112, 672, 112, 3, 1, True, h_swish),
                Bneck(112, 672, 160, 5, 2, True, h_swish),
                Bneck(160, 960, 160, 5, 1, True, h_swish),
                Bneck(160, 960, 160, 5, 1, True, h_swish),
            )

            self.classify = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, bias=False),
                nn.BatchNorm2d(960), h_swish,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(960, 1280), h_swish,
                nn.Linear(1280, num_classes))

        else:
            self.blocks = nn.Sequential(
                Bneck(16, 16, 16, 3, 2, True, relu),
                Bneck(16, 72, 24, 3, 2, False, relu),
                Bneck(24, 88, 24, 3, 1, False, relu),
                Bneck(24, 96, 40, 5, 2, True, h_swish),
                Bneck(40, 240, 40, 5, 1, True, h_swish),
                Bneck(40, 240, 40, 5, 1, True, h_swish),
                Bneck(40, 120, 48, 5, 1, True, h_swish),
                Bneck(48, 144, 48, 5, 1, True, h_swish),
                Bneck(48, 288, 96, 5, 2, True, h_swish),
                Bneck(96, 576, 96, 5, 1, True, h_swish),
                Bneck(96, 576, 96, 5, 1, True, h_swish)
            )

            self.classify = nn.Sequential(
                nn.Conv2d(96, 576, kernel_size=1, bias=False),
                nn.BatchNorm2d(576), h_swish,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(576, 1024), h_swish,
                nn.Linear(1024, num_classes))

    def forward(self, x):
        y = self.stem(x)
        y = self.blocks(y)
        y = self.classify(y)
        return y


mm = utils.ModelManager(MobileNetV3(1, 10, False), "../../weights/mobilenetv3.pt")
dataloader = utils.load_fashion_mnist(batch_size=64, size=224, train=True)
mm.train(dataloader, nn.CrossEntropyLoss(), 5)
mm.test(dataloader, utils.score_acc)

dataloader = utils.load_fashion_mnist(batch_size=64, size=224, train=False)
mm.test(dataloader, utils.score_acc)

mm.save("../../weights/mobilenetv3.pt")
