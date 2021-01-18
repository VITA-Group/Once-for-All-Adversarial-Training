import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def linf_clamp(x, _min, _max):
    '''
    Inplace linf clamping on Tensor x.

    Args:
        x: Tensor. shape=(N,C,W,H)
        _min: Tensor with same shape as x.
        _max: Tensor with same shape as x.
    '''
    idx = x.data < _min
    x.data[idx] = _min[idx]
    idx = x.data > _max
    x.data[idx] = _max[idx]

    return x

class PGD():
    def __init__(self, eps, steps=7, alpha=None, loss_fn=None, targeted=False, use_FiLM=False):
        '''
        Args:
            eps: float. noise bound.
            steps: int. PGD attack step number.
            alpha: float. step size for PGD attack.
            loss_fn: loss function which is maximized to generate adversarial images.
            targeted: bool. If Ture, do targeted attack.
        '''
        self.steps = steps
        self.eps = eps
        self.alpha = alpha if alpha else min(eps * 1.25, eps + 4/255) / steps 
        self.targeted = targeted
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction="sum")    
        self.use_FiLM = use_FiLM   


    def attack(self, model, x, labels=None, targets=None, _lambda=None, idx2BN=None):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            labels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        # 
        model.eval().cuda()

        # initialize x_adv:
        x_adv = x.clone()
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        x_adv = Variable(x_adv.cuda(), requires_grad=True)

        for t in range(self.steps):
            if self.use_FiLM:
                logits_adv = model(x_adv, _lambda=_lambda, idx2BN=idx2BN)
            else:
                logits_adv = model(x_adv)
            if self.targeted:
                loss_adv = - self.loss_fn(logits_adv, targets)
            else: # untargeted attack
                loss_adv = self.loss_fn(logits_adv, labels)
            grad_adv = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            x_adv.data.add_(self.alpha * torch.sign(grad_adv.data)) # gradient assend by Sign-SGD
            x_adv = linf_clamp(x_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
            x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
            
        return x_adv


if __name__ == "__main__":
    import os
    from collections import OrderedDict
    from tqdm import tqdm

    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    from models.cifar10.resnet import ResNet34 
    from utils.context import ctx_noparamgrad_and_eval
    from utils.utils import *
    from attacks.pgd import PGD
    from attacks.all_attackers import get_all_attackers

    from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack, DDNL2Attack

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    torch.backends.cudnn.benchmark = True
    
    # data:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = CIFAR10('datasets/cifar10', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    # model:
    model = ResNet34().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(
        os.path.join('results_cifar10_ResNet', 
        'cifar10_ResNet_untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.5',
        'latest.pth')))

    # compare our PGD attacker with torchvision PGD attacker:
    attacker_pgd = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
        nb_iter=7, eps_iter=8/255*1.25/7, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
    attacker_dict = [
        ('pgd_ours', PGD(eps=8/255, steps=7)),
        ('pgd', attacker_pgd)
    ]

    # test model:
    model.eval()
    test_accs, test_accs_adv = AverageMeter(), AverageMeter()
    SA_dict, RA_dict = OrderedDict(), OrderedDict()
    for attacker_name, attacker in attacker_dict:
        print('evaluating using %s...' % attacker_name)
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            imgs, labels = imgs.cuda(), labels.cuda()
            with ctx_noparamgrad_and_eval(model):
                if isinstance(attacker, PGD) :
                    imgs_adv = attacker.attack(x=imgs, model=model, labels=labels)  
                else:
                    imgs_adv = attacker.perturb(imgs, labels) 
            linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
            logits_adv = model(imgs_adv.detach())
            logits = model(imgs)
            # accs:
            test_accs.append((logits.argmax(1) == labels).float().mean().item())
            test_accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        # 
        print('%s: RA %.4f, SA: %.4f' % (attacker_name, test_accs_adv.avg, test_accs.avg))

        SA_dict[attacker_name] = test_accs.avg
        RA_dict[attacker_name] = test_accs_adv.avg
    print(SA_dict)
    print(RA_dict)
