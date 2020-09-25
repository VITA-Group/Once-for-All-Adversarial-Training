import torch
import torch.nn as nn
import torch.nn.functional as F


class WrapOATModel(nn.Module):
    def __init__(self, model, _lambda, idx2BN):
        super(WrapOATModel, self).__init__()
        self.model = model
        self._lambda = _lambda
        self.idx2BN = idx2BN
        
    def forward(self, x):
        return self.model(x, self._lambda, self.idx2BN)