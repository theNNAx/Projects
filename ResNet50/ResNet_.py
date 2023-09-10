import torch
import numpy as np
import torch.cuda

from torch import nn


class BasicBlock(nn.Module):
    expasion = 1

    def __init__(self, in_ch, block_ch, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, block_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(block_ch, block_ch*self.expasion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_ch*self.expasion)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        print("After dowunsample:",identity.shape)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        print(identity.shape)
        out += identity
        return self.relu2(out)


class Bottleneck(nn.Module):
    expasion = 4

    def __init__(self, in_ch, block_ch, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, block_ch, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_ch, block_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_ch)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(block_ch, block_ch*self.expasion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(block_ch*self.expasion)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
            # print("After dowunsample:", identity.shape)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        # print(out.shape)
        out += identity
        return self.relu2(out)


class ResNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, block=Bottleneck, block_num=[3, 4, 6, 3]):
        super().__init__()
        self.in_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_ch = 64

        self.layer1 = self._make_layer(block, 64, block_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        self.fc_layer = nn.Sequential(
            nn.Linear(512*block.expasion*2*2, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.maxpool1(self.bn1(self.conv1(x)))  # (64, 3, 32 32) -> (64, 64, 16, 16)
        out = self.layer1(out)  # (64, 64, 16, 16) -> (64, 64, 16, 16)
        out = self.layer2(out)  # (64, 64, 16, 16)  -> (64, 128, 8, 8)
        out = self.layer3(out)  # (64, 128, 8, 8)  -> (64, 256, 4, 4)
        out = self.layer4(out)  # (64, 256, 8, 8)  -> (64, 512, 2, 2)

        out = out.reshape(out.shape[0], -1)  # 将数据拉平
        out = self.fc_layer(out) # (64, 512, 2, 2)  -> (64*1*10)
        return out

    def _make_layer(self, block, block_ch, block_num, stride=1):
        layers = []
        downsample = nn.Conv2d(self.in_ch, block_ch*block.expasion, kernel_size=1, stride=stride)
        layers += [block(self.in_ch, block_ch, downsample=downsample, stride=stride)]
        self.in_ch = block_ch*block.expasion

        for _ in range(1, block_num):
            layers += [block(self.in_ch, block_ch)]
        return nn.Sequential(*layers)


def data_tf(x):

    x = np.array(x, dtype='float32')
    # x = (x - 0.5) / 0.5  # 标准化
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维
    x = torch.from_numpy(x)
    return x






