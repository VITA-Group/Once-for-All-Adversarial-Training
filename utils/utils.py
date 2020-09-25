import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:
        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model(normalized_input)


def requires_grad_(model:nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)


class LambdaLR():
	def __init__(self, n_epochs, offset, decay_start_epoch):
		assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
		self.n_epochs = n_epochs
		self.offset = offset
		self.decay_start_epoch = decay_start_epoch

	def step(self, epoch):
		return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def soft_sign(w, th):
	'''
	pytorch soft-sign function

    Args:
        w: Tensor
        th: float
	'''
	with torch.no_grad():
		temp = torch.abs(w) - th
		# print('th:', th)
		# print('temp:', temp.size())
		return torch.sign(w) * nn.functional.relu(temp)


def save_ckpt(epoch, model, optimizer, scheduler, best_TA, best_ATA, training_loss, val_TA, val_ATA, path):
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_TA': best_TA,
        'best_ATA': best_ATA,
        'training_loss': training_loss,
        'val_TA': val_TA,
        'val_ATA': val_ATA,
    }
    torch.save(ckpt, path)

def load_ckpt(model, optimizer, scheduler, path):
    if not os.path.isfile(path):
        raise Exception('No such file: %s' % path)
    print("===>>> loading checkpoint from %s" % path)
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    best_TA = ckpt['best_TA']
    best_ATA = ckpt['best_ATA']
    training_loss = ckpt['training_loss']
    val_TA = ckpt['val_TA']
    val_ATA = ckpt['val_ATA']
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA


def fourD2threeD(batch, n_row=10):
	'''
	Convert a batch of images (N,W,H,C) to a single big image (W*n, H*m, C)
	Input:
		batch: type=ndarray, shape=(N,W,H,C)
	Return:
		rows: type=ndarray, shape=(W*n, H*m, C)
	'''
	N = batch.shape[0]
	img_list = np.split(batch, N)
	for i, img in enumerate(img_list):
		img_list[i] = img.squeeze(axis=0)
	one_row = np.concatenate(img_list, axis=1)
	# print('one_row:', one_row.shape)
	row_list = np.split(one_row, n_row, axis=1)
	rows = np.concatenate(row_list, axis=0)
	return rows

from torch.autograd import Variable
from torch.nn.functional import normalize
def SRIP(model):
    '''
    https://github.com/htwang14/CycleGANCompression/commit/c312603345f2efdd4c090399d4730de2a1a1c10b
    '''
	
    for W in model.parameters():
        if W.ndimension() < 2:
            continue
        else:
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1,cols)
            wt = torch.transpose(w1,0,1)
            ident = Variable(torch.eye(cols,cols)).cuda()

            w_tmp = (torch.matmul(wt,w1) - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            
            l2_reg = (torch.norm(sigma,2))**2
			
    return l2_reg
