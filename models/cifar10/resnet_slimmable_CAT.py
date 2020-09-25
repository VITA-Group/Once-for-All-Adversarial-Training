''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.slimmable_ops import SlimmableConv2d, SlimmableLinear, width_mult_list, SwitchableBatchNorm2d
from models.FiLM import SlimmableFiLM_Layer
from models.DualBN import SwitchableDualBN2d

class SlimmableBasicBlockOAT(nn.Module):

    def __init__(self, in_planes_list, out_planes_list, stride=1, use2BN=True, FiLM_in_channels=1):
        super(SlimmableBasicBlockOAT, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = SwitchableDualBN2d
        else:
            Norm2d = SwitchableBatchNorm2d

        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn1 = Norm2d(out_planes_list)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = SlimmableConv2d(out_planes_list, out_planes_list, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2 = Norm2d(out_planes_list)

        if stride != 1 or list(in_planes_list) != list(out_planes_list):
            self.mismatch = True
            self.conv_sc = SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(out_planes_list)
        else:
            self.mismatch = False

        self.film1 = SlimmableFiLM_Layer(channels_list=out_planes_list, in_channels=FiLM_in_channels) 
        self.film2 = SlimmableFiLM_Layer(channels_list=out_planes_list, in_channels=FiLM_in_channels)


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


class SlimmableResNet34OAT(nn.Module):
    def __init__(self, num_classes=10, FiLM_in_channels=1, use2BN=True):
        super(SlimmableResNet34OAT, self).__init__()
        self.use2BN = use2BN

        list64 = np.array([int(64 * width_mult) for width_mult in width_mult_list])
        list128 = np.array([int(128 * width_mult) for width_mult in width_mult_list])
        list256 = np.array([int(256 * width_mult) for width_mult in width_mult_list])
        list512 = np.array([int(512 * width_mult) for width_mult in width_mult_list])


        self.conv1 = SlimmableConv2d([3 for _ in width_mult_list], list64,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        if self.use2BN:
            self.bn1 = SwitchableDualBN2d(list64)
        else:
            self.bn1 = SwitchableBatchNorm2d(list64)
        self.film1 = SlimmableFiLM_Layer(list64, in_channels=FiLM_in_channels)


        self.bundle1 = nn.ModuleList([
            SlimmableBasicBlockOAT(list64, list64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list64, list64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list64, list64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle2 = nn.ModuleList([
            SlimmableBasicBlockOAT(list64, list128, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list128, list128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list128, list128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list128, list128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle3 = nn.ModuleList([
            SlimmableBasicBlockOAT(list128, list256, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list256, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list256, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list256, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list256, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list256, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle4 = nn.ModuleList([
            SlimmableBasicBlockOAT(list256, list512, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list512, list512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBasicBlockOAT(list512, list512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        

        self.linear = SlimmableLinear(list512, [num_classes for width_mult in width_mult_list])
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


class SlimmableBottleneckBlockOAT(nn.Module):

    def __init__(self, in_planes_list, out_planes_list, stride=1, use2BN=True, FiLM_in_channels=1):
        super(SlimmableBottleneckBlockOAT, self).__init__()

        # print('in_planes_lst:', in_planes_list, type(in_planes_list))
        # print('out_planes_lst:', out_planes_list, type(out_planes_list))

        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = SwitchableDualBN2d
        else:
            Norm2d = SwitchableBatchNorm2d

        self.conv1 = SlimmableConv2d(in_planes_list, out_planes_list, kernel_size=1, bias=False)
        self.bn1 = Norm2d(out_planes_list)
        self.conv2 = SlimmableConv2d(out_planes_list, out_planes_list, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = Norm2d(out_planes_list)
        self.conv3 = SlimmableConv2d(out_planes_list, 4*out_planes_list, kernel_size=1, bias=False)
        self.bn3 = Norm2d(4*out_planes_list)

        if stride != 1 or list(in_planes_list) != list(4*out_planes_list):
            self.mismatch = True
            self.conv_sc = SlimmableConv2d(in_planes_list, 4*out_planes_list, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(4*out_planes_list)
        else:
            self.mismatch = False

        self.film1 = SlimmableFiLM_Layer(channels_list=out_planes_list, in_channels=FiLM_in_channels) 
        self.film2 = SlimmableFiLM_Layer(channels_list=out_planes_list, in_channels=FiLM_in_channels)
        self.film3 = SlimmableFiLM_Layer(channels_list=4*out_planes_list, in_channels=FiLM_in_channels)


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
        out = F.relu(out)

        out = self.conv3(out)
        if self.use2BN:
            out = self.bn3(out, idx2BN)
        else:
            out = self.bn3(out)
        out = self.film3(out, _lambda)

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


class SlimmableResNet50OAT(nn.Module):
    def __init__(self, num_classes=10, FiLM_in_channels=1, use2BN=True):
        super(SlimmableResNet50OAT, self).__init__()
        self.use2BN = use2BN

        list64 = np.array([int(64 * width_mult) for width_mult in width_mult_list])
        list128 = np.array([int(128 * width_mult) for width_mult in width_mult_list])
        list256 = np.array([int(256 * width_mult) for width_mult in width_mult_list])
        list512 = np.array([int(512 * width_mult) for width_mult in width_mult_list])


        self.conv1 = SlimmableConv2d(np.array([3 for _ in width_mult_list]), list64,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        if self.use2BN:
            self.bn1 = SwitchableDualBN2d(list64)
        else:
            self.bn1 = SwitchableBatchNorm2d(list64)
        self.film1 = SlimmableFiLM_Layer(list64, in_channels=FiLM_in_channels)


        self.bundle1 = nn.ModuleList([
            SlimmableBottleneckBlockOAT(list64, list64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list64*4, list64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list64*4, list64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle2 = nn.ModuleList([
            SlimmableBottleneckBlockOAT(list64*4, list128, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list128*4, list128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list128*4, list128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list128*4, list128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle3 = nn.ModuleList([
            SlimmableBottleneckBlockOAT(list128*4, list256, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list256*4, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list256*4, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list256*4, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list256*4, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list256*4, list256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle4 = nn.ModuleList([
            SlimmableBottleneckBlockOAT(list256*4, list512, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list512*4, list512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            SlimmableBottleneckBlockOAT(list512*4, list512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        

        self.linear = SlimmableLinear(list512 * 4, np.array([num_classes for width_mult in width_mult_list]))
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
    # model = SlimmableResNet34OAT()
    model = SlimmableResNet50OAT()
    for width_mult in sorted(width_mult_list, reverse=True):
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        # for name, m in model.named_modules():
        #     if isinstance(m, SlimmableLinear) or isinstance(m, SlimmableConv2d) or isinstance(m, SwitchableDualBN2d):
        #         print(name, m.width_mult)