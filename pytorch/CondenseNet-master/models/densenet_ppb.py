from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from layers import Conv
from torch.nn.modules.utils import _pair
from torch.nn.modules.padding import ConstantPad3d

__all__ = ['DenseNetPPB']


def make_divisible(x, y):
    return int((x // y + 1) * y) if x % y else int(x)

class prefpoolb(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, kernel_size=2, padding=0, 
            bias=True, width=32, height=32, num_convs = 4):
        super(prefpoolb, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.kernel_size = _pair(kernel_size)
        self.stride = stride #replacing stride with reduction operation
        self.padding = _pair(padding)
        self.width = width
        self.height = height
        self.bias = bias
        self.nc = num_convs

        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self.maxpref = nn.Parameter(torch.Tensor(1,in_planes,1,1))
        self.pconvs = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size, num_convs))
        self.pprefs = nn.Parameter(torch.Tensor(out_planes,1,1,1,num_convs))
        if bias:
            self.mb = nn.Parameter(torch.Tensor(1))
            self.pb = nn.Parameter(torch.Tensor(num_convs))
        else:
            self.register_parameter('bias', None)
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     n = self.out_channels*(self.kernel_size[0]**2)
    #     stdv = 1. / math.sqrt(n)
    #     self.yw.data.uniform_(-ystdv, ystdv)
    #     self.yb.data.uniform_(-xstdv, xstdv)
    #     if self.yb is not None:
    #         self.yb.data.uniform_(-ystdv, ystdv)
    #         self.xb.data.uniform_(-xstdv, xstdv)

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
        out = (self.mb+self.maxpref)*self.maxpool(x)
        for c in range(self.nc):
            pconv = self.pconvs.select(4, c)
            pconv = pconv.repeat(self.out_channels,1,1,1)
            ppref = self.pb.select(0, c) + self.pprefs.select(4, c)
            pconv = (ppref)+pconv
            out += F.conv2d(x, pconv, None, self.stride, self.padding, groups = self.in_channels)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = Conv(in_channels, args.bottleneck * growth_rate,
                           kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, args, width, height):
        super(_Transition, self).__init__()
        self.conv = Conv(in_channels, out_channels,
                         kernel_size=1, groups=args.group_1x1)
        self.pool = prefpoolb(out_channels, out_channels, 
                             stride=2, width=width, height=height)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNetPPB(nn.Module):
    def __init__(self, args):

        super(DenseNetPPB, self).__init__()

        self.width = 32
        self.height = 32
        self.stages = args.stages
        self.growth = args.growth
        self.reduction = args.reduction
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Set initial width to 2 x growth_rate[0]
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, prefpoolb):
                n = m.kernel_size[0] * m.kernel_size[1]
                m.maxpref.data.normal_(0, math.sqrt(2. / (m.in_channels)))
                m.pconvs.data.normal_(0, math.sqrt(2. / (n*m.nc)))
                m.pprefs.data.normal_(0, math.sqrt(2. / (m.in_channels*m.nc)))
                m.mb.data.zero_()
                m.pb.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            out_features = make_divisible(math.ceil(self.num_features * self.reduction),
                                          self.args.group_1x1)
            trans = _Transition(in_channels=self.num_features,
                                out_channels=out_features,
                                args=self.args, 
                                width=self.width//(2**(i)), 
                                height=self.height//(2**(i)))
            self.features.add_module('transition_%d' % (i + 1), trans)
            self.num_features = out_features
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            ### Use adaptive ave pool as global pool
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
