import torch

from torch import nn
from torch.nn import functional as F

from .op import FusedLeakyReLU



class OffsetsFeatureShiftConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activate = FusedLeakyReLU(out_channels)
    
    def forward(self, input):
        """input: feature tensor either from the generator intermediate layer or from FSE"""
        return self.activate(self.conv(input))


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            groups=groups,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            groups=groups,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class FeatureiResnet(nn.Module):
    def __init__(self, blocks, inplanes=1024):
        super().__init__()

        self.res_blocks = {}

        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block

            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, bias=False), nn.BatchNorm2d(planes, eps=1e-05))

                self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes

        self.res_blocks = nn.ModuleDict(self.res_blocks)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x