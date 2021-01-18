''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FiLM import FiLM_Layer
from models.DualBN import DualBN2d

class BasicBlockOAT(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, use2BN=False, FiLM_in_channels=1):
        super(BasicBlockOAT, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm2d(out_planes)

        if stride != 1 or in_planes != out_planes:
            self.mismatch = True
            self.conv_sc = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(out_planes)
        else:
            self.mismatch = False
        
        self.film1 = FiLM_Layer(channels=mid_planes, in_channels=FiLM_in_channels) 
        self.film2 = FiLM_Layer(channels=out_planes, in_channels=FiLM_in_channels)

    def forward(self, x, _lambda, idx2BN=None):
        out = self.conv1(x)
        if self.use2BN:
            out = self.bn1(out, idx2BN)
        else:
            out = self.bn1(out)
        out = self.film1(out, _lambda)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use2BN:
            out = self.bn2(out, idx2BN)
        else:
            out = self.bn2(out)
        out = self.film2(out, _lambda)
        if self.mismatch:
            if self.use2BN: 
                out += self.bn_sc(self.conv_sc(x), idx2BN)
            else:
                out += self.bn_sc(self.conv_sc(x))
        else:
            out += x
        out = F.relu(out)
        # print(out.size())
        return out



class ResNet34OAT(nn.Module):
    '''
    GFLOPS: 1.1837, model size: 31.4040MB
    '''
    def __init__(self, num_classes=10, FiLM_in_channels=1, use2BN=False):
        super(ResNet34OAT, self).__init__()
        self.use2BN = use2BN

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use2BN:
            self.bn1 = DualBN2d(64)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.film1 = FiLM_Layer(channels=64, in_channels=FiLM_in_channels)
        self.bundle1 = nn.ModuleList([
            BasicBlockOAT(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle2 = nn.ModuleList([
            BasicBlockOAT(64, 128, 128, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle3 = nn.ModuleList([
            BasicBlockOAT(128, 256, 256, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle4 = nn.ModuleList([
            BasicBlockOAT(256, 512, 512, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(512, 512, 512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(512, 512, 512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.linear = nn.Linear(512, num_classes)
        self.bundles = [self.bundle1, self.bundle2, self.bundle3, self.bundle4]

    def forward(self, x, _lambda, idx2BN=None):
        out = self.conv1(x)
        if self.use2BN:
            out = self.bn1(out, idx2BN)
        else:
            out = self.bn1(out)
        out = self.film1(out, _lambda)
        out = F.relu(out)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out, _lambda, idx2BN)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    from thop import profile
    net = ResNet34OAT()
    x = torch.randn(1,3,32,32)
    _lambda = torch.ones(1,1)
    flops, params = profile(net, inputs=(x, _lambda))
    y = net(x, _lambda)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))