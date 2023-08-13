# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import builtins
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.sampler import LabeledDataSampler
from data.imagenet import *

import backbone as backbone_models
from models.simmatchv2 import get_simmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
import data.transforms as data_transforms
from engine import validate


backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                    help='path to train annotation_u (default: None)')
parser.add_argument('--arch', metavar='ARCH', default='SimMatchV2',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--port', default=23456, type=int, help='dist init port')                    
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to supervised pretrained model (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--anno-percent', type=float, default=0.1,
                    help='number of labeled data')
parser.add_argument('--split-seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mu', default=5, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.7, type=float, help='pseudo label threshold')
parser.add_argument('--eman', action='store_true', default=False, help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float, help='EMA decay rate')
parser.add_argument('--norm', default='None', type=str, help='the normalization for backbone (default: None)')
parser.add_argument('--moco-path', default=None, type=str)
parser.add_argument('--model-prefix', default='encoder_q', type=str, help='the model prefix of self-supervised pretrained state_dict')


parser.add_argument('--t', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--da_m', type=float, default=0.9)  
parser.add_argument('--k', type=int, default=256)
parser.add_argument('--topn', type=int, default=128)
parser.add_argument('--lambda_nn', type=float, default=10)
parser.add_argument('--lambda_ee', type=float, default=5)
parser.add_argument('--lambda_ne', type=float, default=5)
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--norm_feat', default=False, action='store_true')
parser.add_argument('--multicrop', default=False, action='store_true')
args = parser.parse_args()


def main_worker():
    best_acc1 = 0
    best_acc5 = 0

    rank, local_rank, world_size = dist_utils.dist_init(args.port)
    args.gpu = local_rank
    args.rank = rank
    args.world_size = world_size
    args.distributed = True

    if rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    print(args)
    
    train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl()
    label_bank = torch.tensor([label  for (label, _) in train_dataset_x.samples]).long()

    # Data loading code
    train_sampler = DistributedSampler

    train_loader_u = DataLoader(
        train_dataset_u,
        sampler=train_sampler(train_dataset_u),
        batch_size=args.batch_size * args.mu, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader_x = DataLoader(
        train_dataset_x,
        sampler=LabeledDataSampler(train_dataset_x, num_samples=len(train_loader_u) * args.batch_size),
        batch_size=args.batch_size, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        sampler=train_sampler(val_dataset),
        batch_size=64, shuffle=False, persistent_workers=True,
        num_workers=args.workers, pin_memory=True)
    

    # create model
    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_simmatch_model(args.arch)
    norm = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm,
        K=args.k * args.batch_size * args.mu * args.world_size,
        label_bank=label_bank,
        args=args
    )


    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    checkpoint_path = args.checkpoint
    best_checkpoint_path = args.checkpoint[:-4] + '_best.pth'
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        best_acc5 = checkpoint['best_acc5']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True

    if args.evaluate:
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if rank == 0:
            print('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))

    else:
        for epoch in range(args.start_epoch, args.epochs):
            if epoch >= args.warmup_epoch:
                lr_schedule.adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader_x, train_loader_u, model, optimizer, epoch, scaler, args)

            if (epoch + 1) % args.eval_freq == 0:
                # evaluate on validation set
                acc1, acc5 = validate(val_loader, model, criterion, args)
                # remember best acc@1 and save checkpoint
                best_acc1 = max(acc1, best_acc1)
                best_acc5 = max(acc5, best_acc5)

            if rank == 0:
                print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, acc1, acc5, best_acc1, best_acc5))
                torch.save({
                    'epoch': epoch + 1,
                    'best_acc1': float(best_acc1),
                    'best_acc5': float(best_acc5),
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, checkpoint_path)

                if best_acc1 == acc1:
                    torch.save({
                        'epoch': epoch + 1,
                        'best_acc1': float(best_acc1),
                        'best_acc5': float(best_acc5),
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                    }, best_checkpoint_path)


def train(train_loader_x, train_loader_u, model, optimizer, epoch, scaler, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_sup = utils.AverageMeter('Loss_sup', ':.4e')
    losses_nn = utils.AverageMeter('Loss_nn', ':.4e')
    losses_ee = utils.AverageMeter('Loss_ee', ':.4e')
    losses_ne = utils.AverageMeter('Loss_ne', ':.4e')
    curr_lr = utils.InstantMeter('LR', ':.4e')
    progress = utils.ProgressMeter(
        len(train_loader_u),
        [curr_lr, batch_time, data_time, losses, losses_sup, losses_nn, losses_ee, losses_ne],
        prefix="Epoch: [{}/{}]\t".format(epoch, args.epochs))

    epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        train_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        train_loader_u.sampler.set_epoch(epoch)

    train_iter_x = iter(train_loader_x)
    # switch to train mode
    model.train()
    if args.eman:
        print("setting the ema model to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()
    for i, (images_u, targets_u) in enumerate(train_loader_u):
        try:
            images_x, targets_x, index = next(train_iter_x)
        except Exception:
            epoch_x += 1
            print("reshuffle train_loader_x at epoch={}".format(epoch_x))
            if args.distributed:
                print("set epoch={} for labeled sampler".format(epoch_x))
                train_loader_x.sampler.set_epoch(epoch_x)
            train_iter_x = iter(train_loader_x)
            images_x, targets_x, index = next(train_iter_x)

        images_u_w, images_u_s = images_u[0], images_u[1:]

        # measure data loading time
        data_time.update(time.time() - end)


        images_x = images_x.cuda(args.gpu, non_blocking=True)
        images_u_w = images_u_w.cuda(args.gpu, non_blocking=True)
        
        for img_idx in range(len(images_u_s)):
            images_u_s[img_idx] = images_u_s[img_idx].cuda(args.gpu, non_blocking=True)
            
        targets_x = targets_x.cuda(args.gpu, non_blocking=True)
        targets_u = targets_u.cuda(args.gpu, non_blocking=True)
        index = index.cuda(args.gpu, non_blocking=True)

        totlal_step = args.epochs * len(train_loader_u)
        curr_step = epoch * len(train_loader_u) + i + 1
        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader_u)
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # model forward
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_qx, logits_qu, loss_ee, loss_ne, prob_ku, pseudo_label = model(images_x, images_u_w, images_u_s, index, args)
            max_probs, _ = torch.max(prob_ku, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            loss_sup = F.cross_entropy(logits_qx, targets_x, reduction='mean')
            loss_nn = (torch.sum(-F.log_softmax(logits_qu, dim=1) * pseudo_label.detach(), dim=1) * mask.detach()).mean()
            loss = loss_sup + args.lambda_nn * loss_nn + args.lambda_ee * loss_ee + args.lambda_ne * loss_ne

        # measure accuracy and record loss
        losses.update(loss.item())
        losses_sup.update(loss_sup.item())
        losses_nn.update(loss_nn.item())
        losses_ee.update(loss_ee.item())
        losses_ne.update(loss_ne.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema_m = args.ema_m if args.ema_m > 0.996 else 1 - (1 - args.ema_m) * (math.cos(math.pi * curr_step / float(totlal_step)) + 1) / 2
        # update the ema model
        if hasattr(model, 'module'):
            model.module.momentum_update_ema(ema_m)
        else:
            model.momentum_update_ema(ema_m)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def get_imagenet_ssl():
    weak_transform = data_transforms.weak_aug
    strong_transform = data_transforms.moco_aug

    transform_x = weak_transform
    if args.multicrop:
        transform_u_list = [
            weak_transform,
            data_transforms.get_dino_aug(size=224, scale=(0.140, 1.000), gaussian=0.5, solarize=0.1),
            data_transforms.get_dino_aug(size=192, scale=(0.117, 0.860), gaussian=0.5, solarize=0.1),
            data_transforms.get_dino_aug(size=160, scale=(0.095, 0.715), gaussian=0.5, solarize=0.1),
            data_transforms.get_dino_aug(size=120, scale=(0.073, 0.571), gaussian=0.5, solarize=0.1),
            data_transforms.get_dino_aug(size=96 , scale=(0.050, 0.429), gaussian=0.5, solarize=0.1),
        ]
    else:
        transform_u_list = [
            weak_transform,
            strong_transform
        ]
        
    transform_u = data_transforms.MultiTransform(transform_u_list)
    transform_val = data_transforms.eval_aug

    train_dataset_x = ImagenetPercent(percent=args.anno_percent, labeled=True, aug=transform_x, return_index=True)
    train_dataset_u = ImagenetPercent(percent=args.anno_percent, labeled=False, aug=transform_u)
    val_dataset = Imagenet(mode='val', aug=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset



if __name__ == '__main__':
    main_worker()
