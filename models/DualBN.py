import torch
import torch.nn as nn
from models.slimmable_ops import width_mult_list, SlimmableBatchNorm2d

class DualBN2d(nn.Module):
    '''
    Element wise Dual BN, efficient implementation. (Boolean tensor indexing is very slow!)
    '''
    def __init__(self, num_features):
        '''
        Args:
            num_features: int. Number of channels
        '''
        super(DualBN2d, self).__init__()
        self.BN_c = nn.BatchNorm2d(num_features)
        self.BN_a = nn.BatchNorm2d(num_features)

    def forward(self, _input, idx):
        '''
        Args:
            _input: Tensor. size=(N,C,H,W)
            idx: int. _input[0:idx,...] -> _lambda=0 -> BN_c; _input[idx:,...] -> _lambda!=0 -> BN_a
        
        Returns:
            _output: Tensor. size=(N,C,H,W)
        '''
        if idx == 0:
            _output = self.BN_a(_input)
        elif idx == _input.size()[0]:
            _output = self.BN_c(_input)
        else:
            _output_c = self.BN_c(_input[0:idx,...]) # BN cannot take tensor with N=0
            _output_a = self.BN_a(_input[idx:,...])
            _output = torch.cat([_output_c, _output_a], dim=0)
        return _output


class SwitchableDualBN2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableDualBN2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list) # total channel number.
        BNs_c, BNs_a = [], []
        for i in num_features_list:
            BNs_c.append(nn.BatchNorm2d(i))
            BNs_a.append(nn.BatchNorm2d(i))
        self.BN_c = nn.ModuleList(BNs_c)
        self.BN_a = nn.ModuleList(BNs_a)
        self.width_mult = max(width_mult_list) # change this attribute using model.apply() during runtime.
        self.ignore_model_profiling = True

    def forward(self, _input, sample_idx):
        BN_idx = width_mult_list.index(self.width_mult)
        if sample_idx == 0:
            _output = self.BN_a[BN_idx](_input)
        elif sample_idx == _input.size()[0]:
            _output = self.BN_c[BN_idx](_input)
        else:
            _output_c = self.BN_c[BN_idx](_input[0:sample_idx,...]) # BN cannot take tensor with N=0
            _output_a = self.BN_a[BN_idx](_input[sample_idx:,...])
            _output = torch.cat([_output_c, _output_a], dim=0)
        return _output

   
class SlimmableDualBN2d(nn.Module):
    '''
    DualBN2d shared by all sub-networks in slimmable network.
    This shouldn't work according to slimmable net paper.
    '''
    def __init__(self, num_features_list):
        '''
        Args:
            num_features_list: [int]. Number of channels for each sub-network
        '''
        super(SlimmableDualBN2d, self).__init__()
        self.BN_c = SlimmableBatchNorm2d(num_features_list)
        self.BN_a = SlimmableBatchNorm2d(num_features_list)

    def forward(self, _input, idx):
        '''
        Args:
            _input: Tensor. size=(N,C,H,W)
            idx: int. _input[0:idx,...] -> _lambda=0 -> BN_c; _input[idx:,...] -> _lambda!=0 -> BN_a
        
        Returns:
            _output: Tensor. size=(N,C,H,W)
        '''
        if idx == 0:
            _output = self.BN_a(_input)
        elif idx == _input.size()[0]:
            _output = self.BN_c(_input)
        else:
            _output_c = self.BN_c(_input[0:idx,...]) # BN cannot take tensor with N=0
            _output_a = self.BN_a(_input[idx:,...])
            _output = torch.cat([_output_c, _output_a], dim=0)
        return _output

