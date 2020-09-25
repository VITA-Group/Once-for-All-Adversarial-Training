import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from models.slimmable_ops import width_mult_list, SlimmableLinear

class FiLM_Layer(nn.Module):
    def __init__(self, channels, in_channels=1, alpha=1, activation=F.leaky_relu):
        '''
        input size: (N, in_channels). output size: (N, channels)
        
        Args:
            channels: int.
            alpha: scalar. Expand ratio for FiLM hidden layer.
        '''
        super(FiLM_Layer, self).__init__()
        self.channels = channels
        self.activation = activation
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, alpha*channels*2, bias=True), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(alpha*channels*2, channels*2, bias=True), 
        )
        
    def forward(self, _input, _lambda):
        N, C, H, W = _input.size()
        out = self.MLP(_lambda)
        self.mu, self.sigma = torch.split(out, [self.channels, self.channels], dim=-1)
        if self.activation is not None:
            self.mu, self.sigma = self.activation(self.mu), self.activation(self.sigma)
        _output = _input * self.mu.view(N, C, 1, 1).expand_as(_input) + self.sigma.view(N, C, 1, 1).expand_as(_input)
        return _output


class SlimmableFiLM_Layer(nn.Module):
    def __init__(self, channels_list, in_channels=1, alpha=1, activation=F.leaky_relu):
        '''
        Args:
            channels_list: [int]
            alpha: scalar. Expand ratio for FiLM hidden layer.
        '''
        super(SlimmableFiLM_Layer, self).__init__()
        self.activation = activation
        self.MLP = nn.Sequential(
            SlimmableLinear([in_channels for _ in width_mult_list], alpha*np.array(channels_list)*2, bias=True), 
            nn.LeakyReLU(inplace=True),
            SlimmableLinear(alpha*np.array(channels_list)*2, np.array(channels_list)*2, bias=True), 
        )
        
    def forward(self, _input, _lambda):
        N, C, H, W = _input.size()
        out = self.MLP(_lambda)
        mu, sigma = torch.split(out, [C, C], dim=-1)
        if self.activation is not None:
            mu, sigma = self.activation(mu), self.activation(sigma)
        _output = _input * mu.view(N, C, 1, 1).expand_as(_input) + sigma.view(N, C, 1, 1).expand_as(_input)
        return _output

