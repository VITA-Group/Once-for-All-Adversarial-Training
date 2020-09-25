''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from models.slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d, SlimmableLinear
from models.slimmable_ops import width_mult_list

class SlimmableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes_lst, out_planes_lst, stride=1):
        super(SlimmableBasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = SlimmableConv2d(in_planes_lst, out_planes_lst, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn1 = SwitchableBatchNorm2d(out_planes_lst)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = SlimmableConv2d(out_planes_lst, out_planes_lst, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2 = SwitchableBatchNorm2d(out_planes_lst)

        self.shortcut = nn.Sequential()
        if stride != 1 or list(in_planes_lst) != list(out_planes_lst):
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                SlimmableConv2d(in_planes_lst, out_planes_lst, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_planes),
                SwitchableBatchNorm2d(out_planes_lst),
            )


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        return out


class SlimmableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes_lst, out_planes_lst, stride=1):
        super(SlimmableBottleneck, self).__init__()
        
        # print('in_planes_lst:', in_planes_lst, type(in_planes_lst))
        # print('out_planes_lst:', out_planes_lst, type(out_planes_lst))

        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = SlimmableConv2d(in_planes_lst, out_planes_lst, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SwitchableBatchNorm2d(out_planes_lst)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = SlimmableConv2d(out_planes_lst, out_planes_lst, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SwitchableBatchNorm2d(out_planes_lst)
        # self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 = SlimmableConv2d(out_planes_lst, self.expansion*out_planes_lst, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.bn3 = SwitchableBatchNorm2d(self.expansion*out_planes_lst)

        self.shortcut = nn.Sequential()
        if stride != 1 or list(in_planes_lst) != list(self.expansion*out_planes_lst):
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                SlimmableConv2d(in_planes_lst, self.expansion*out_planes_lst, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
                SwitchableBatchNorm2d(self.expansion*out_planes_lst),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        return out


class SlimmableResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SlimmableResNet, self).__init__()
        self.in_planes_list = np.array([int(64 * width_mult) for width_mult in width_mult_list])

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SlimmableConv2d(np.array([3 for _ in width_mult_list]), self.in_planes_list,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = SwitchableBatchNorm2d(self.in_planes_list)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(block, np.array([int(64 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer2 = self._make_layer(block, np.array([int(128 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer3 = self._make_layer(block, np.array([int(256 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4 = self._make_layer(block, np.array([int(512* width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[3], stride=2)
        # self.linear = nn.Linear(512, num_classes)
        self.linear = SlimmableLinear(
            np.array([int(512* width_mult) for width_mult in width_mult_list]) * block.expansion, 
            np.array([num_classes for width_mult in width_mult_list])
        )

    def _make_layer(self, block, planes_list, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_list, planes_list, stride))
            self.in_planes_list = planes_list * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SlimmableResNet34():
    '''
    GFLOPS: , model size: 
    '''
    return SlimmableResNet(SlimmableBasicBlock, [3,4,6,3])

def SlimmableResNet50():
    '''
    GFLOPS: , model size: 
    '''
    return SlimmableResNet(SlimmableBottleneck, [3,4,6,3])



if __name__ == '__main__':
    # model = SlimmableResNet34()
    model = SlimmableResNet50()
    for width_mult in sorted(width_mult_list, reverse=True):
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        # for name, m in model.named_modules():
        #     if isinstance(m, SlimmableLinear) or isinstance(m, SlimmableConv2d) or isinstance(m, SwitchableBatchNorm2d):
        #         print(name, m.width_mult)