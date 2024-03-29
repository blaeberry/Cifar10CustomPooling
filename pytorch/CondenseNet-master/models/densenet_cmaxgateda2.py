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

__all__ = ['DenseNetCmaxGatedA2']


def make_divisible(x, y):
    return int((x // y + 1) * y) if x % y else int(x)

class cmaxga2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, kernel_size=2, padding=0, 
            bias=True, width=32, height=32, levels=2):
        super(cmaxga2, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.kernel_size = _pair(kernel_size)
        self.stride = stride #replacing stride with reduction operation
        self.padding = _pair(padding)
        self.width = width
        self.height = height
        self.bias = bias
        self.levels = levels
        self.leaves = levels**2

        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self.maxgate = nn.Parameter(torch.Tensor(out_planes, 1, kernel_size, kernel_size))
        self.pconvs = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size, self.leaves))
        self.pgates = nn.Parameter(torch.Tensor(out_planes, 1, kernel_size, kernel_size, (self.leaves+levels)//2))
        if bias:
            self.mb = nn.Parameter(torch.Tensor(out_planes))
            self.pbs = nn.Parameter(torch.Tensor(out_planes, self.leaves))
            self.gbs = nn.Parameter(torch.Tensor(out_planes, (self.leaves+levels)//2))
        else:
            self.register_parameter('bias', None)

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
        max_out = self.maxpool(x)
        # depthwise convolutions
        max_gate = F.conv2d(x, self.maxgate, self.mb, self.stride, self.padding, groups = self.in_channels)
        out = max_out*max_gate
        leaves = []
        for c in range(self.leaves//2):
            pconv1 = self.pconvs.select(4, c*2).contiguous()
            pconv2 = self.pconvs.select(4, (c*2)+1).contiguous()
            pgate = self.pgates.select(4, c).contiguous()
            pconv1 = pconv1.repeat(self.out_channels,1,1,1)
            pconv2 = pconv2.repeat(self.out_channels,1,1,1)
            gate_out = F.conv2d(x, pgate, self.gbs.select(1, c).contiguous(), self.stride, self.padding, groups = self.in_channels)
            gate_out = F.sigmoid(gate_out)
            pool1_out =  F.conv2d(x, pconv1, self.pbs.select(1, c*2).contiguous(), self.stride, self.padding, groups = self.in_channels)
            pool2_out =  F.conv2d(x, pconv2, self.pbs.select(1, (c*2)+1).contiguous(), self.stride, self.padding, groups = self.in_channels)
            leaves.append(gate_out*pool1_out)
            leaves.append((1.0 - gate_out)*pool2_out)
        nodes = []
        for n in range(self.leaves//2):
            nodes.append(leaves[n*2]+leaves[n*2+1])
        for n in range(len(nodes)//2):
            pgate = self.pgates.select(4, (self.leaves//2)+n).contiguous()
            gate_out = F.conv2d(F.pad(nodes[n], (0,1,0,1)), pgate, self.gbs.select(1, (self.leaves//2)+n).contiguous(), 1, self.padding, groups = self.in_channels)
            gate_out = F.sigmoid(gate_out)
            out += (nodes[n*2]*gate_out) + (nodes[(n*2)+1]*(1.0-gate_out))
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
        self.pool = cmaxga2(out_channels, out_channels, 
                             stride=2, width=width, height=height)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNetCmaxGatedA2(nn.Module):
    def __init__(self, args):

        super(DenseNetCmaxGatedA2, self).__init__()

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
            if isinstance(m, cmaxga2):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.pgates.data.normal_(0, math.sqrt(2. / n))
                m.maxgate.data.normal_(0, math.sqrt(2. / n))
                m.pconvs.data.fill_(1)
                m.mb.data.zero_()
                m.pbs.data.zero_()
                m.gbs.data.zero_()
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
