import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.modules.padding import ConstantPad3d

import sys
import numpy as np
import math

__all__ = ['Wide_ResNet_2D_G_Resize_Avg_Shuffle']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        if hasattr(m, 'chpref'):
            init.xavier_uniform(m.chpref, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

#need to make sure that class has 'Conv' in the name
class ConvCust(nn.Module):
    def __init__(self, in_planes, out_planes, groups, stride=1, kernel_size=3, padding=0, bias=True):
        super(ConvCust, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_planes, 1, *self.kernel_size))
        self.chpref = nn.Parameter(torch.Tensor(out_planes, in_planes//groups, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        stdv2 = 1. / math.sqrt(self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        self.chpref.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        conv = self.weight*self.chpref #get normal conv dimensions
        return F.conv2d(x, conv, self.bias, self.stride, self.padding, groups=self.groups)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, groups, stride=1):
        super(wide_basic, self).__init__()
        self.groups = groups
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ConvCust(in_planes, planes, groups, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ConvCust(planes, planes, groups, kernel_size=3, stride=stride, padding=1, bias=True)

        sequence = []
        if stride != 1: 
            sequence.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        if in_planes != planes:
            extra = planes-in_planes
            sequence.append(ConstantPad3d((0, 0, 0, 0, 0, extra), 0))
        self.shortcut = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = channel_shuffle(out, self.groups)
        out = self.conv2(F.relu(self.bn2(out)))
        out = channel_shuffle(out, self.groups)
        out += channel_shuffle(channel_shuffle(self.shortcut(x), self.groups), self.groups)

        return out

class Wide_ResNet_2D_G_Resize_Avg_Shuffle(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, groups):
        super(Wide_ResNet_2D_G_Resize_Avg_Shuffle, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, groups=groups)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, groups=groups)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, groups=groups)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, groups):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, groups, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet_2D_G_Resize_Avg_Shuffle(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
