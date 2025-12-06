from typing import Type

import torch.nn as nn


# https://arxiv.org/pdf/1512.03385
# The standard color augmentation in [21] is used. We adopt batch
# normalization (BN) [16] right after each convolution and
# before activation, following [16].
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        downsample=None,
    ):
        """_Basic block definition as per original ResNet paper_

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            stride (int, optional): _description_. Defaults to 1.
        """
        super().__init__()

        self.resnet_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = downsample
        self.ReLU = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.resnet_block(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.ReLU(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks: list[int],
        block: Type[nn.Module] | None = None,
        num_classes: int = 10,
        channels: int = 1,
        in_channels: int = 64,
    ):
        """_ResNet module very similar to the one originally used for the
        CIFAR-10 dataset with some minor changes._


        Args:
            num_blocks (list[int]): _description_
            block (Type[nn.Module] | None, optional): _Block definition used
            for the layers; can technically be any module but BasicBlock for
            now_. If `None` is provided, BasicBlock will be used
            num_classes (int, optional): _Number of classes_. Defaults to 10.
            channels (int, optional): _Number of channels; usually corresponds
            to number of color channels _. Defaults to 1.
            in_channels (int, optional): _Number of initial filters produced at
            the first convolutional layer_. Defaults to 64.
        """
        super().__init__()
        self.in_channels = int(in_channels * 0.5)

        if block is None:
            block = BasicBlock

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                int(in_channels * 0.5),
                kernel_size=3,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(int(in_channels * 0.5)),
        )
        self.layer1 = self._make_layer(
            block, int(in_channels * 0.5), num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, int(in_channels * 1), num_blocks[1], stride=2
        )
        # self.layer3 = self._make_layer(block, int(in_channels * 2), num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, int(in_channels * 4), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            int(
                in_channels * 1,
            ),
            num_classes,
        )
        # self.linear = nn.Linear(int(in_channels * 4,), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # [stride, 1, 1, ... N] -> only first block downsamples
        downsample = None
        if stride != 1 or self.in_channels != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(
            block(self.in_channels, planes, stride=stride, downsample=downsample)
        )
        self.in_channels = planes
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, out):
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
