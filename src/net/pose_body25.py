import torch
from torch import nn
from collections import OrderedDict
import numpy as np


def load_dict_from_npy(weight_file):
    map_from_np2ts = {'conv1_1': 'blk00.conv00',
                      'conv1_2': 'blk00.conv01',
                      'conv2_1': 'blk00.conv10',
                      'conv2_2': 'blk00.conv11',
                      'conv3_1': 'blk00.conv20',
                      'conv3_2': 'blk00.conv21',
                      'conv3_3': 'blk00.conv22',
                      'conv3_4': 'blk00.conv23',
                      'conv4_1': 'blk00.conv30',
                      'conv4_2': 'blk00.conv31',
                      'prelu4_2': 'blk00.prelu31',
                      'conv4_3_CPM': 'blk00.conv32',
                      'prelu4_3_CPM': 'blk00.prelu32',
                      'conv4_4_CPM': 'blk00.conv33',
                      'prelu4_4_CPM': 'blk00.prelu33',
                      'Mconv1_stage0_L2_0': 'blk10.sub0.conv0',
                      'Mprelu1_stage0_L2_0': 'blk10.sub0.prelu0',
                      'Mconv1_stage0_L2_1': 'blk10.sub0.conv1',
                      'Mprelu1_stage0_L2_1': 'blk10.sub0.prelu1',
                      'Mconv1_stage0_L2_2': 'blk10.sub0.conv2',
                      'Mprelu1_stage0_L2_2': 'blk10.sub0.prelu2',
                      'Mconv2_stage0_L2_0': 'blk10.sub1.conv0',
                      'Mprelu2_stage0_L2_0': 'blk10.sub1.prelu0',
                      'Mconv2_stage0_L2_1': 'blk10.sub1.conv1',
                      'Mprelu2_stage0_L2_1': 'blk10.sub1.prelu1',
                      'Mconv2_stage0_L2_2': 'blk10.sub1.conv2',
                      'Mprelu2_stage0_L2_2': 'blk10.sub1.prelu2',
                      'Mconv3_stage0_L2_0': 'blk10.sub2.conv0',
                      'Mprelu3_stage0_L2_0': 'blk10.sub2.prelu0',
                      'Mconv3_stage0_L2_1': 'blk10.sub2.conv1',
                      'Mprelu3_stage0_L2_1': 'blk10.sub2.prelu1',
                      'Mconv3_stage0_L2_2': 'blk10.sub2.conv2',
                      'Mprelu3_stage0_L2_2': 'blk10.sub2.prelu2',
                      'Mconv4_stage0_L2_0': 'blk10.sub3.conv0',
                      'Mprelu4_stage0_L2_0': 'blk10.sub3.prelu0',
                      'Mconv4_stage0_L2_1': 'blk10.sub3.conv1',
                      'Mprelu4_stage0_L2_1': 'blk10.sub3.prelu1',
                      'Mconv4_stage0_L2_2': 'blk10.sub3.conv2',
                      'Mprelu4_stage0_L2_2': 'blk10.sub3.prelu2',
                      'Mconv5_stage0_L2_0': 'blk10.sub4.conv0',
                      'Mprelu5_stage0_L2_0': 'blk10.sub4.prelu0',
                      'Mconv5_stage0_L2_1': 'blk10.sub4.conv1',
                      'Mprelu5_stage0_L2_1': 'blk10.sub4.prelu1',
                      'Mconv5_stage0_L2_2': 'blk10.sub4.conv2',
                      'Mprelu5_stage0_L2_2': 'blk10.sub4.prelu2',
                      'Mconv6_stage0_L2': 'blk10.conv0',
                      'Mprelu6_stage0_L2': 'blk10.prelu0',
                      'Mconv7_stage0_L2': 'blk10.conv1',
                      'Mconv1_stage1_L2_0': 'blk11.sub0.conv0',
                      'Mprelu1_stage1_L2_0': 'blk11.sub0.prelu0',
                      'Mconv1_stage1_L2_1': 'blk11.sub0.conv1',
                      'Mprelu1_stage1_L2_1': 'blk11.sub0.prelu1',
                      'Mconv1_stage1_L2_2': 'blk11.sub0.conv2',
                      'Mprelu1_stage1_L2_2': 'blk11.sub0.prelu2',
                      'Mconv2_stage1_L2_0': 'blk11.sub1.conv0',
                      'Mprelu2_stage1_L2_0': 'blk11.sub1.prelu0',
                      'Mconv2_stage1_L2_1': 'blk11.sub1.conv1',
                      'Mprelu2_stage1_L2_1': 'blk11.sub1.prelu1',
                      'Mconv2_stage1_L2_2': 'blk11.sub1.conv2',
                      'Mprelu2_stage1_L2_2': 'blk11.sub1.prelu2',
                      'Mconv3_stage1_L2_0': 'blk11.sub2.conv0',
                      'Mprelu3_stage1_L2_0': 'blk11.sub2.prelu0',
                      'Mconv3_stage1_L2_1': 'blk11.sub2.conv1',
                      'Mprelu3_stage1_L2_1': 'blk11.sub2.prelu1',
                      'Mconv3_stage1_L2_2': 'blk11.sub2.conv2',
                      'Mprelu3_stage1_L2_2': 'blk11.sub2.prelu2',
                      'Mconv4_stage1_L2_0': 'blk11.sub3.conv0',
                      'Mprelu4_stage1_L2_0': 'blk11.sub3.prelu0',
                      'Mconv4_stage1_L2_1': 'blk11.sub3.conv1',
                      'Mprelu4_stage1_L2_1': 'blk11.sub3.prelu1',
                      'Mconv4_stage1_L2_2': 'blk11.sub3.conv2',
                      'Mprelu4_stage1_L2_2': 'blk11.sub3.prelu2',
                      'Mconv5_stage1_L2_0': 'blk11.sub4.conv0',
                      'Mprelu5_stage1_L2_0': 'blk11.sub4.prelu0',
                      'Mconv5_stage1_L2_1': 'blk11.sub4.conv1',
                      'Mprelu5_stage1_L2_1': 'blk11.sub4.prelu1',
                      'Mconv5_stage1_L2_2': 'blk11.sub4.conv2',
                      'Mprelu5_stage1_L2_2': 'blk11.sub4.prelu2',
                      'Mconv6_stage1_L2': 'blk11.conv0',
                      'Mprelu6_stage1_L2': 'blk11.prelu0',
                      'Mconv7_stage1_L2': 'blk11.conv1',
                      'Mconv1_stage2_L2_0': 'blk12.sub0.conv0',
                      'Mprelu1_stage2_L2_0': 'blk12.sub0.prelu0',
                      'Mconv1_stage2_L2_1': 'blk12.sub0.conv1',
                      'Mprelu1_stage2_L2_1': 'blk12.sub0.prelu1',
                      'Mconv1_stage2_L2_2': 'blk12.sub0.conv2',
                      'Mprelu1_stage2_L2_2': 'blk12.sub0.prelu2',
                      'Mconv2_stage2_L2_0': 'blk12.sub1.conv0',
                      'Mprelu2_stage2_L2_0': 'blk12.sub1.prelu0',
                      'Mconv2_stage2_L2_1': 'blk12.sub1.conv1',
                      'Mprelu2_stage2_L2_1': 'blk12.sub1.prelu1',
                      'Mconv2_stage2_L2_2': 'blk12.sub1.conv2',
                      'Mprelu2_stage2_L2_2': 'blk12.sub1.prelu2',
                      'Mconv3_stage2_L2_0': 'blk12.sub2.conv0',
                      'Mprelu3_stage2_L2_0': 'blk12.sub2.prelu0',
                      'Mconv3_stage2_L2_1': 'blk12.sub2.conv1',
                      'Mprelu3_stage2_L2_1': 'blk12.sub2.prelu1',
                      'Mconv3_stage2_L2_2': 'blk12.sub2.conv2',
                      'Mprelu3_stage2_L2_2': 'blk12.sub2.prelu2',
                      'Mconv4_stage2_L2_0': 'blk12.sub3.conv0',
                      'Mprelu4_stage2_L2_0': 'blk12.sub3.prelu0',
                      'Mconv4_stage2_L2_1': 'blk12.sub3.conv1',
                      'Mprelu4_stage2_L2_1': 'blk12.sub3.prelu1',
                      'Mconv4_stage2_L2_2': 'blk12.sub3.conv2',
                      'Mprelu4_stage2_L2_2': 'blk12.sub3.prelu2',
                      'Mconv5_stage2_L2_0': 'blk12.sub4.conv0',
                      'Mprelu5_stage2_L2_0': 'blk12.sub4.prelu0',
                      'Mconv5_stage2_L2_1': 'blk12.sub4.conv1',
                      'Mprelu5_stage2_L2_1': 'blk12.sub4.prelu1',
                      'Mconv5_stage2_L2_2': 'blk12.sub4.conv2',
                      'Mprelu5_stage2_L2_2': 'blk12.sub4.prelu2',
                      'Mconv6_stage2_L2': 'blk12.conv0',
                      'Mprelu6_stage2_L2': 'blk12.prelu0',
                      'Mconv7_stage2_L2': 'blk12.conv1',
                      'Mconv1_stage3_L2_0': 'blk13.sub0.conv0',
                      'Mprelu1_stage3_L2_0': 'blk13.sub0.prelu0',
                      'Mconv1_stage3_L2_1': 'blk13.sub0.conv1',
                      'Mprelu1_stage3_L2_1': 'blk13.sub0.prelu1',
                      'Mconv1_stage3_L2_2': 'blk13.sub0.conv2',
                      'Mprelu1_stage3_L2_2': 'blk13.sub0.prelu2',
                      'Mconv2_stage3_L2_0': 'blk13.sub1.conv0',
                      'Mprelu2_stage3_L2_0': 'blk13.sub1.prelu0',
                      'Mconv2_stage3_L2_1': 'blk13.sub1.conv1',
                      'Mprelu2_stage3_L2_1': 'blk13.sub1.prelu1',
                      'Mconv2_stage3_L2_2': 'blk13.sub1.conv2',
                      'Mprelu2_stage3_L2_2': 'blk13.sub1.prelu2',
                      'Mconv3_stage3_L2_0': 'blk13.sub2.conv0',
                      'Mprelu3_stage3_L2_0': 'blk13.sub2.prelu0',
                      'Mconv3_stage3_L2_1': 'blk13.sub2.conv1',
                      'Mprelu3_stage3_L2_1': 'blk13.sub2.prelu1',
                      'Mconv3_stage3_L2_2': 'blk13.sub2.conv2',
                      'Mprelu3_stage3_L2_2': 'blk13.sub2.prelu2',
                      'Mconv4_stage3_L2_0': 'blk13.sub3.conv0',
                      'Mprelu4_stage3_L2_0': 'blk13.sub3.prelu0',
                      'Mconv4_stage3_L2_1': 'blk13.sub3.conv1',
                      'Mprelu4_stage3_L2_1': 'blk13.sub3.prelu1',
                      'Mconv4_stage3_L2_2': 'blk13.sub3.conv2',
                      'Mprelu4_stage3_L2_2': 'blk13.sub3.prelu2',
                      'Mconv5_stage3_L2_0': 'blk13.sub4.conv0',
                      'Mprelu5_stage3_L2_0': 'blk13.sub4.prelu0',
                      'Mconv5_stage3_L2_1': 'blk13.sub4.conv1',
                      'Mprelu5_stage3_L2_1': 'blk13.sub4.prelu1',
                      'Mconv5_stage3_L2_2': 'blk13.sub4.conv2',
                      'Mprelu5_stage3_L2_2': 'blk13.sub4.prelu2',
                      'Mconv6_stage3_L2': 'blk13.conv0',
                      'Mprelu6_stage3_L2': 'blk13.prelu0',
                      'Mconv7_stage3_L2': 'blk13.conv1',
                      'Mconv1_stage0_L1_0': 'blk20.sub0.conv0',
                      'Mprelu1_stage0_L1_0': 'blk20.sub0.prelu0',
                      'Mconv1_stage0_L1_1': 'blk20.sub0.conv1',
                      'Mprelu1_stage0_L1_1': 'blk20.sub0.prelu1',
                      'Mconv1_stage0_L1_2': 'blk20.sub0.conv2',
                      'Mprelu1_stage0_L1_2': 'blk20.sub0.prelu2',
                      'Mconv2_stage0_L1_0': 'blk20.sub1.conv0',
                      'Mprelu2_stage0_L1_0': 'blk20.sub1.prelu0',
                      'Mconv2_stage0_L1_1': 'blk20.sub1.conv1',
                      'Mprelu2_stage0_L1_1': 'blk20.sub1.prelu1',
                      'Mconv2_stage0_L1_2': 'blk20.sub1.conv2',
                      'Mprelu2_stage0_L1_2': 'blk20.sub1.prelu2',
                      'Mconv3_stage0_L1_0': 'blk20.sub2.conv0',
                      'Mprelu3_stage0_L1_0': 'blk20.sub2.prelu0',
                      'Mconv3_stage0_L1_1': 'blk20.sub2.conv1',
                      'Mprelu3_stage0_L1_1': 'blk20.sub2.prelu1',
                      'Mconv3_stage0_L1_2': 'blk20.sub2.conv2',
                      'Mprelu3_stage0_L1_2': 'blk20.sub2.prelu2',
                      'Mconv4_stage0_L1_0': 'blk20.sub3.conv0',
                      'Mprelu4_stage0_L1_0': 'blk20.sub3.prelu0',
                      'Mconv4_stage0_L1_1': 'blk20.sub3.conv1',
                      'Mprelu4_stage0_L1_1': 'blk20.sub3.prelu1',
                      'Mconv4_stage0_L1_2': 'blk20.sub3.conv2',
                      'Mprelu4_stage0_L1_2': 'blk20.sub3.prelu2',
                      'Mconv5_stage0_L1_0': 'blk20.sub4.conv0',
                      'Mprelu5_stage0_L1_0': 'blk20.sub4.prelu0',
                      'Mconv5_stage0_L1_1': 'blk20.sub4.conv1',
                      'Mprelu5_stage0_L1_1': 'blk20.sub4.prelu1',
                      'Mconv5_stage0_L1_2': 'blk20.sub4.conv2',
                      'Mprelu5_stage0_L1_2': 'blk20.sub4.prelu2',
                      'Mconv6_stage0_L1': 'blk20.conv0',
                      'Mprelu6_stage0_L1': 'blk20.prelu0',
                      'Mconv7_stage0_L1': 'blk20.conv1',
                      'Mconv1_stage1_L1_0': 'blk21.sub0.conv0',
                      'Mprelu1_stage1_L1_0': 'blk21.sub0.prelu0',
                      'Mconv1_stage1_L1_1': 'blk21.sub0.conv1',
                      'Mprelu1_stage1_L1_1': 'blk21.sub0.prelu1',
                      'Mconv1_stage1_L1_2': 'blk21.sub0.conv2',
                      'Mprelu1_stage1_L1_2': 'blk21.sub0.prelu2',
                      'Mconv2_stage1_L1_0': 'blk21.sub1.conv0',
                      'Mprelu2_stage1_L1_0': 'blk21.sub1.prelu0',
                      'Mconv2_stage1_L1_1': 'blk21.sub1.conv1',
                      'Mprelu2_stage1_L1_1': 'blk21.sub1.prelu1',
                      'Mconv2_stage1_L1_2': 'blk21.sub1.conv2',
                      'Mprelu2_stage1_L1_2': 'blk21.sub1.prelu2',
                      'Mconv3_stage1_L1_0': 'blk21.sub2.conv0',
                      'Mprelu3_stage1_L1_0': 'blk21.sub2.prelu0',
                      'Mconv3_stage1_L1_1': 'blk21.sub2.conv1',
                      'Mprelu3_stage1_L1_1': 'blk21.sub2.prelu1',
                      'Mconv3_stage1_L1_2': 'blk21.sub2.conv2',
                      'Mprelu3_stage1_L1_2': 'blk21.sub2.prelu2',
                      'Mconv4_stage1_L1_0': 'blk21.sub3.conv0',
                      'Mprelu4_stage1_L1_0': 'blk21.sub3.prelu0',
                      'Mconv4_stage1_L1_1': 'blk21.sub3.conv1',
                      'Mprelu4_stage1_L1_1': 'blk21.sub3.prelu1',
                      'Mconv4_stage1_L1_2': 'blk21.sub3.conv2',
                      'Mprelu4_stage1_L1_2': 'blk21.sub3.prelu2',
                      'Mconv5_stage1_L1_0': 'blk21.sub4.conv0',
                      'Mprelu5_stage1_L1_0': 'blk21.sub4.prelu0',
                      'Mconv5_stage1_L1_1': 'blk21.sub4.conv1',
                      'Mprelu5_stage1_L1_1': 'blk21.sub4.prelu1',
                      'Mconv5_stage1_L1_2': 'blk21.sub4.conv2',
                      'Mprelu5_stage1_L1_2': 'blk21.sub4.prelu2',
                      'Mconv6_stage1_L1': 'blk21.conv0',
                      'Mprelu6_stage1_L1': 'blk21.prelu0',
                      'Mconv7_stage1_L1': 'blk21.conv1'}

    try:
        np_weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        np_weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    ts_weights_dict = {}

    for k, v in np_weights_dict.items():
        ts_k = map_from_np2ts[k]
        if 'conv' in k:
            ts_weights_dict[ts_k + '.weight'] = torch.from_numpy(v['weights'])
            ts_weights_dict[ts_k + '.bias'] = torch.from_numpy(v['bias']).flatten()
        else:
            ts_weights_dict[ts_k + '.weight'] = torch.from_numpy(v['weights'])
    return ts_weights_dict


class SubBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(SubBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.prelu0 = nn.PReLU(num_parameters=growth_rate)
        self.conv1 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=growth_rate)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU(num_parameters=growth_rate)

    def forward(self, x):
        y1 = self.prelu0(self.conv0(x))
        y2 = self.prelu1(self.conv1(y1))
        y3 = self.prelu2(self.conv2(y2))
        return torch.cat((y1, y2, y3), dim=1)


def block(in_channels, growth_rate, hidden_channels, out_channels):
    midway_in_channels = 3 * growth_rate
    return nn.Sequential(OrderedDict(
        sub0=SubBlock(in_channels, growth_rate),
        sub1=SubBlock(midway_in_channels, growth_rate),
        sub2=SubBlock(midway_in_channels, growth_rate),
        sub3=SubBlock(midway_in_channels, growth_rate),
        sub4=SubBlock(midway_in_channels, growth_rate),
        conv0=nn.Conv2d(midway_in_channels, hidden_channels, 1),
        prelu0=nn.PReLU(num_parameters=hidden_channels),
        conv1=nn.Conv2d(hidden_channels, out_channels, 1)))


'''
x   ->  b00 -   -   -   -   -   -   -   -   -   -
        |       |       |       |       |       |
        |   ->  b10 ->  b11 ->  b12 ->  b13 -   -   -   -
                                        |       |       |
                                        b20 ->  b21 ->  y
'''


class PoseBody25(nn.Module):
    def __init__(self):
        super(PoseBody25, self).__init__()
        self.blk00 = nn.Sequential(OrderedDict(
            conv00=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), relu00=nn.ReLU(),
            conv01=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), relu01=nn.ReLU(),
            maxpool0=nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1, 0, 1)),
            conv10=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), relu10=nn.ReLU(),
            conv11=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), relu11=nn.ReLU(),
            maxpool1=nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1, 0, 1)),
            conv20=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), relu20=nn.ReLU(),
            conv21=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu21=nn.ReLU(),
            conv22=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu22=nn.ReLU(),
            conv23=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu23=nn.ReLU(),
            maxpool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 1, 0, 1)),
            conv30=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), relu30=nn.ReLU(),
            conv31=nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            prelu31=nn.PReLU(num_parameters=512),
            conv32=nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            prelu32=nn.PReLU(num_parameters=256),
            conv33=nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            prelu33=nn.PReLU(num_parameters=128))
        )
        self.blk10 = block(128, 96, 256, 52)
        self.blk11 = block(180, 128, 512, 52)
        self.blk12 = block(180, 128, 512, 52)
        self.blk13 = block(180, 128, 512, 52)
        self.blk20 = block(180, 96, 256, 26)
        self.blk21 = block(206, 128, 512, 26)

    def forward(self, x):
        # layer 0
        y0 = self.blk00(x)

        # layer 1
        y1 = self.blk10(y0)
        x1 = torch.cat((y0, y1), 1)
        y1 = self.blk11(x1)
        x1 = torch.cat((y0, y1), 1)
        y1 = self.blk12(x1)
        x1 = torch.cat((y0, y1), 1)
        y1 = self.blk13(x1)

        # layer 2
        x2 = torch.cat((y0, y1), 1)
        y2 = self.blk20(x2)
        x2 = torch.cat((y0, y2, y1), 1)
        y2 = self.blk21(x2)

        y = torch.cat((y2, y1), 1)
        return y


if __name__ == '__main__':
    model = PoseBody25()
    model.load_state_dict(load_dict_from_npy('../../model/pose_body_25_weight.npy'))
