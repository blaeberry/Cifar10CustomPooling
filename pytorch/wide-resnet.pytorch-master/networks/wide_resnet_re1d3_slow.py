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

__all__ = ['Wide_ResNet_RE1D3_Slow']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'cw'):
            init.constant(m.cb, 0)
            init.xavier_uniform(m.cw, gain=np.sqrt(2))
        if hasattr(m, 'weight'):
            init.xavier_uniform(m.weight, gain=np.sqrt(2))
            init.constant(m.bias, 0)
        if hasattr(m, 'yw'):
            init.xavier_uniform(m.yw, gain=np.sqrt(2))
            init.xavier_uniform(m.xw, gain=np.sqrt(2))
            init.constant(m.yb, 0)
            init.constant(m.xb, 0)
    elif classname.find('BatchNorm') != -1:
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

class ConvCust1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=1, padding=0, 
                bias=True, width=32, height=32):
        super(ConvCust1, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.kernel_size = _pair(kernel_size)
        self.stride = stride #replacing stride with reduction operation
        self.padding = _pair(padding)
        self.width = width
        self.height = height
        self.cw = nn.Parameter(torch.Tensor(out_planes, in_planes, 1, 1))
        self.yw = nn.Parameter(torch.Tensor(int(height*stride), height, 1, 1))
        self.xw = nn.Parameter(torch.Tensor(int(width*stride), width, 1, 1))
        if bias:
            self.cb = nn.Parameter(torch.Tensor(out_planes))
            self.yb = nn.Parameter(torch.Tensor(int(height*stride)))
            self.xb = nn.Parameter(torch.Tensor(int(width*stride)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        # for k in self.kernel_size:
        #     n *= k
        cstdv = 1. / math.sqrt(self.in_channels)
        ystdv = 1. / math.sqrt(self.height)
        xstdv = 1. / math.sqrt(self.width)
        self.cw.data.uniform_(-cstdv, cstdv)
        self.yw.data.uniform_(-ystdv, ystdv)
        self.yb.data.uniform_(-xstdv, xstdv)
        if self.cb is not None:
            self.cb.data.uniform_(-cstdv, cstdv)
            self.yb.data.uniform_(-ystdv, ystdv)
            self.xb.data.uniform_(-xstdv, xstdv)

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
        x = F.conv2d(x, self.cw, self.cb, 1, self.padding)
        x = x.permute(0, 2, 3, 1).contiguous() #N H W C
        x = F.conv2d(x, self.yw, self.yb, 1, self.padding)
        x = x.permute(0, 2, 3, 1).contiguous() #N W C H
        x = F.conv2d(x, self.xw, self.xb, 1, self.padding)
        x = x.permute(0, 2, 3, 1).contiguous() #N C H W
        return x


class ConvCust(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=1, padding=0, 
                bias=True, width=32, height=32):
        super(ConvCust, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.kernel_size = _pair(kernel_size)
        self.stride = stride #replacing stride with reduction operation
        self.padding = _pair(padding)
        self.width = width
        self.height = height
        self.yw = nn.Parameter(torch.Tensor(int(height*stride), height, 1, 1))
        self.xw = nn.Parameter(torch.Tensor(int(width*stride), width, 1, 1))
        if bias:
            self.yb = nn.Parameter(torch.Tensor(int(height*stride)))
            self.xb = nn.Parameter(torch.Tensor(int(width*stride)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        ystdv = 1. / math.sqrt(self.height)
        xstdv = 1. / math.sqrt(self.width)
        self.yw.data.uniform_(-ystdv, ystdv)
        self.yb.data.uniform_(-xstdv, xstdv)
        if self.yb is not None:
            self.yb.data.uniform_(-ystdv, ystdv)
            self.xb.data.uniform_(-xstdv, xstdv)

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
        x = x.permute(0, 2, 3, 1).contiguous() #N H W C
        x = F.conv2d(x, self.yw, self.yb, 1, self.padding)
        x = x.permute(0, 2, 3, 1).contiguous() #N W C H
        x = F.conv2d(x, self.xw, self.xb, 1, self.padding)
        x = x.permute(0, 2, 3, 1).contiguous() #N C H W
        return x

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, width=32, height=32):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.custpool = nn.Sequential()
        if stride != 1:
            self.custpool = nn.Sequential(ConvCust1(planes, planes, kernel_size=1, stride=stride, padding=0, 
                                bias=True, width=width, height=height),
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ConvCust1(in_planes, planes, kernel_size=1, stride=stride, padding=0,
                            bias=True, width=width, height=height)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.custpool(out)
        out += self.shortcut(x)

        return out

class Wide_ResNet_RE1D3_Slow(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, width, height):
        super(Wide_ResNet_RE1D3_Slow, self).__init__()
        self.in_planes = 16
        self.width = width
        self.height = height

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, 1, width, height)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, 2, width, height)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, 2, width//2, height//2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, width, height):
        strides =  [1]*(num_blocks)
        if stride != 1:
            strides[0] = 0.75
            strides[1] = 2.0/3.0
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, width, height))
            width = int(width * stride)
            height = int(height * stride)
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
    net=Wide_ResNet_RE1D3_Slow(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
