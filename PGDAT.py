'''
PGDAT
'''

import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet import ResNet34
from models.svhn.wide_resnet import WRN16_8
from models.stl10.wide_resnet import WRN40_2

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from attacks.pgd import PGD

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')
parser.add_argument('--gpu', default='7')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=256, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
# loss parameters:
parser.add_argument('--Lambda', default=0.5, type=float, help='adv loss tradeoff parameter')
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# data loader:
if args.dataset == 'cifar10':
    train_loader, val_loader, _ = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

# model:
if args.dataset == 'cifar10':
    model_fn = ResNet34
elif args.dataset == 'svhn':
    model_fn = WRN16_8
elif args.dataset == 'stl10':
    model_fn = WRN40_2
model = model_fn().cuda()
model = torch.nn.DataParallel(model)

# mkdirs:
model_str = model_fn.__name__
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-%s' % args.decay_epochs
attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%d-%d' % (args.eps, args.steps)
loss_str = 'lambda%s' % (args.Lambda)
save_folder = os.path.join('/hdd1/haotao/OAT_results', args.dataset, model_str, '%s_%s_%s_%s' % (attack_str, opt_str, decay_str, loss_str))
create_dir(save_folder)

# optimizer:
if args.opt == 'sgd':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.opt == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.decay == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
elif args.decay == 'multisteps':
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

# attacker:
attacker = PGD(eps=args.eps/255, steps=args.steps)

# load ckpt:
if args.resume:
    last_epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA \
         = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
    start_epoch = last_epoch + 1
else:
    start_epoch = 0
    best_TA, best_ATA = 0, 0
    # training curve lists:
    training_loss, val_TA, val_ATA = [], [], []

## training:
for epoch in range(start_epoch, args.epochs):
    fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
    start_time = time.time()
    ## training:
    model.train()
    requires_grad_(model, True)
    accs, accs_adv, losses = AverageMeter(), AverageMeter(), AverageMeter()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.cuda(), labels.cuda()

        # generate adversarial images:
        if args.Lambda != 0:
            with ctx_noparamgrad_and_eval(model):
                imgs_adv = attacker.attack(model, imgs, labels)
            logits_adv = model(imgs_adv.detach())
        # logits for clean imgs:
        logits = model(imgs)

        # loss and update:
        loss = F.cross_entropy(logits, labels)
        if args.Lambda != 0:
            loss = (1-args.Lambda) * loss + args.Lambda * F.cross_entropy(logits_adv, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # proximal gradient for channel pruning:
        current_lr = scheduler.get_lr()[0]

        # metrics:
        accs.append((logits.argmax(1) == labels).float().mean().item())
        if args.Lambda != 0:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        losses.append(loss.item())

        if i % 50 == 0:
            train_str = 'Epoch %d-%d | Train | Loss: %.4f, SA: %.4f' % (epoch, i, losses.avg, accs.avg)
            if args.Lambda != 0:
                train_str += ', RA: %.4f' % (accs_adv.avg)
            print(train_str)
    # lr schedualr update at the end of each epoch:
    scheduler.step()


    ## validation:
    model.eval()
    requires_grad_(model, False)
    print(model.training)

    if args.dataset == 'cifar10':
        eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.7*args.epochs)) # boolean
    elif args.dataset == 'svhn':
        eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.25*args.epochs)) # boolean
    
    if eval_this_epoch:
        val_accs, val_accs_adv = AverageMeter(), AverageMeter()
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.cuda(), labels.cuda()

            # generate adversarial images:
            with ctx_noparamgrad_and_eval(model):
                imgs_adv = attacker.attack(model, imgs, labels)
            linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
            logits_adv = model(imgs_adv.detach())
            # logits for clean imgs:
            logits = model(imgs)

            val_accs.append((logits.argmax(1) == labels).float().mean().item())
            val_accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())

        val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f, RA: %.4f, linf: %.4f - %.4f' % (
            epoch, (time.time()-start_time), current_lr, val_accs.avg, val_accs_adv.avg, 
            torch.min(linf_norms).data, torch.max(linf_norms).data)
        print(val_str)
        fp.write(val_str + '\n')

    # save loss curve:
    training_loss.append(losses.avg)
    plt.plot(training_loss)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'training_loss.png'))
    plt.close()

    if eval_this_epoch:
        val_TA.append(val_accs.avg) 
        plt.plot(val_TA, 'r')
        val_ATA.append(val_accs_adv.avg)
        plt.plot(val_ATA, 'g')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_acc.png'))
        plt.close()
    else:
        val_TA.append(val_TA[-1]) 
        plt.plot(val_TA, 'r')
        val_ATA.append(val_ATA[-1])
        plt.plot(val_ATA, 'g')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_acc.png'))
        plt.close()

    # save pth:
    if eval_this_epoch:
        if val_accs.avg >= best_TA:
            best_TA = val_accs.avg
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))
        if val_accs_adv.avg >= best_ATA:
            best_ATA = val_accs_adv.avg
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_RA.pth'))
    save_ckpt(epoch, model, optimizer, scheduler, best_TA, best_ATA, training_loss, val_TA, val_ATA, 
        os.path.join(save_folder, 'latest.pth'))


