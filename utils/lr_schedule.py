# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import math


def adjust_learning_rate_v2(optimizer, epoch, epochs, base_lr, i, iteration_per_epoch, warm_up, start_lr):
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = (base_lr - start_lr)  * T / warmup_iters + start_lr
    else:
        min_lr = base_lr / 1000
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = 0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr) + min_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def warmup_learning_rate(optimizer, curr_step, warmup_step, args):
    """linearly warm up learning rate"""
    lr = args.lr
    scalar = float(curr_step) / float(max(1, warmup_step))
    scalar = min(1., max(0., scalar))
    lr *= scalar
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        progress = float(epoch - args.warmup_epoch) / float(args.epochs - args.warmup_epoch)
        lr *= 0.5 * (1. + math.cos(math.pi * progress))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_with_min(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        min_lr = args.cos_min_lr
        progress = float(epoch - args.warmup_epoch) / float(args.epochs - args.warmup_epoch)
        lr = min_lr + 0.5 * (lr - min_lr) * (1. + math.cos(math.pi * progress))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
