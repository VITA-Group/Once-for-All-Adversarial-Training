import torch
from scipy.stats import uniform, bernoulli
import numpy as np

def element_wise_sample_lambda(sampling, lambda_choices, encoding_mat, batch_size=128, probs=-1):
    '''
    Elementwise sample and encode lambda.
    
    Args:
        sampling: str. sample scheme.
        lambda_choices: list of floats. Useful when sampling is 'disc'.
        encoding_mat: str. encoding scheme. Useful when sampling is 'disc'.
        probs: float. If probs<=0, ignored and use uniform sample.

    Returns:
        _lambda_flat: Tensor. size=(batch_size). For loss.
        _lambda: Tensor. size=(batch_size, 1). FiLM input tensor.
        num_zeros: int. How many 0s are sampled in this batch.
    '''
    if sampling == 'disc':
        lambda_none_zero_choices = list(set(lambda_choices)-set([0]))
        if 0 in lambda_choices:
            if probs > 0:
                num_zeros = np.ceil(probs * batch_size).astype(int)
            else: # uniform sample
                num_zeros = np.ceil(batch_size / len(lambda_choices)).astype(int) 
            # print('num_zeros %d/batch_size %d' % (num_zeros, batch_size))
        else:
            num_zeros = 0
        num_none_zeros = int(batch_size-num_zeros)
        _lambda_zeros = np.zeros((num_zeros,1))
        _lambda_none_zeros = np.random.choice(lambda_none_zero_choices, size=(num_none_zeros,1)) 
        _lambda = np.concatenate([_lambda_zeros,_lambda_none_zeros], axis=0)
        _lambda_flat = np.squeeze(_lambda) / 1.0
        if encoding_mat is not None:
            idx_lst = []
            for lambda_choice in lambda_choices:
                idx_lst.append(np.where(_lambda==lambda_choice)[0])
            for j, idx in enumerate(idx_lst):
                _lambda[idx] = j
            _lambda = _lambda.astype(np.int32)
            _lambda = encoding_mat[_lambda,:]  
    
    
    assert np.amax(_lambda_flat) <= 1 and np.amin(_lambda_flat) >= 0
    _lambda_flat = torch.from_numpy(_lambda_flat).float().cuda() # for loss
    _lambda = torch.from_numpy(_lambda).float().cuda() # for FiLM layers

    return _lambda_flat, _lambda, num_zeros


def batch_wise_sample_lambda(sampling, lambda_choices, encoding_mat, batch_size=128):
    '''
    Batch-wise sample and encode lambda.
    
    Args:
        sampling: str. sample scheme.
        lambda_choices: list of floats. Useful when sampling is 'disc'.
        encoding_mat: str. encoding scheme. Useful when sampling is 'disc'.

    Returns:
        _lambda_flat: Tensor. size=(batch_size). For loss.
        _lambda: Tensor. size=(batch_size, 1). FiLM input tensor.
    '''
    if sampling == 'disc':
        _lambda = np.random.choice(lambda_choices, size=1)
        _lambda = np.tile(_lambda, (batch_size,1)) 
        _lambda_flat = np.squeeze(_lambda) / 1.0
        if encoding_mat is not None:
            idx_lst = []
            for lambda_choice in lambda_choices:
                idx_lst.append(np.where(_lambda==lambda_choice)[0])
            for j, idx in enumerate(idx_lst):
                _lambda[idx] = j
            _lambda = _lambda.astype(np.int32)
            _lambda = encoding_mat[_lambda,:]  
    
    
    assert np.amax(_lambda_flat) <= 1 and np.amin(_lambda_flat) >= 0
    _lambda_flat = torch.from_numpy(_lambda_flat).float().cuda() # for loss
    _lambda = torch.from_numpy(_lambda).float().cuda() # for FiLM layers

    return _lambda_flat, _lambda

