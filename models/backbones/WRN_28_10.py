import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        # return x if self.equalInOut else self.convShortcut(x) + out
        if self.equalInOut:
            return x
        else:
            x = self.convShortcut(x) + out
            return x

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self,  depth=28, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 64*widen_factor*10]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 1, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.maxpool(out1)
        out2 = self.block1(out1)
        out2 = self.maxpool(out2)
        out3 = self.block2(out2)
        out3 = self.maxpool(out3)
        out4 = self.block3(out3)
        out4 = self.maxpool(out4)

        return out4