"""
Training file for training SkipNets for supervised pre-training stage
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import argparse
import time
import logging

import models
import random
from data import *
import scipy.misc
import numpy as np
from functools import reduce
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.optim.optimizer import required
from models.new_resnet import cifar10_rnn_gate_74

def str2bool(s):
    return s.lower() in ['yes', '1', 'true', 'y']

writer = SummaryWriter()
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )

class CusSGD(SGD):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(CusSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

        return

    def step(self, closure=None, fix_no_decay=True):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        if p.requires_grad or (not fix_no_decay):
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH',
                        default='cifar10_feedforward_38',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_feedforward_38)')
    parser.add_argument('--gate-type', type=str, default='ff',
                        choices=['ff', 'rnn'], help='gate type')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset type')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=99999, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=30, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--verbose', action="store_true",
                        help='print layer skipping ratio at training')
    parser.add_argument('--energy', default=1, type=int,
                        help='using energy as regularization term')
    parser.add_argument('--beta', default=1e-5, type=float,
                        help='coefficient')
    parser.add_argument('--minimum', default=100, type=float,
                        help='minimum')
    # Quantization of input, weight, bias and grad
    parser.add_argument('--num_bits', default=8, type=int,
                        help='precision of input/activation')
    parser.add_argument('--num_bits_weight', default=8, type=int,
                        help='precision of weight')
    parser.add_argument('--num_bits_grad', default=32, type=int,
                        help='precision of (layer) gradients')
    parser.add_argument('--biprecision', default=False, type=str2bool,
                        help='use biprecision or not')
    # Predictive (sign) SGD arguments
    parser.add_argument('--predictive_forward', default=False, type=str2bool,
                        help='use predictive net in forward pass')
    parser.add_argument('--predictive_backward', default=True, type=str2bool,
                        help='use predictive net in backward pass')
    parser.add_argument('--msb_bits', default=4, type=int,
                        help='precision of msb part of input')
    parser.add_argument('--msb_bits_weight', default=4, type=int,
                        help='precision of msb part of weight')
    parser.add_argument('--msb_bits_grad', default=16, type=int,
                        help='precision of msb part of (layer) gradient')
    parser.add_argument('--threshold', default=5e-5, type=float,
                        help='threshold to use full precision gradient calculation')
    parser.add_argument('--sparsify', default=False, type=str2bool,
                        help='sparsify the gradients using predictive net method')
    parser.add_argument('--sign', default=True, type=str2bool,
                        help='take sign before applying gradient')
    args = parser.parse_args()
    return args

training_cost = 0
skip_count = 0

def main():
    args = parse_args()

    args.signsgd_config = {
        'num_bits': args.num_bits,
        'num_bits_weight': args.num_bits_weight,
        'num_bits_grad': args.num_bits_grad,
        'biprecision': args.biprecision,
        'predictive_forward': args.predictive_forward,
        'predictive_backward': args.predictive_backward,
        'msb_bits': args.msb_bits,
        'msb_bits_weight': args.msb_bits_weight,
        'msb_bits_grad': args.msb_bits_grad,
        'threshold': args.threshold,
        'sparsify': args.sparsify,
        'sign': args.sign,
    }

    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    os.makedirs(save_path, exist_ok=True)

    # config logger file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)

def translate(dest, src):
    tmp = src['state_dict']['module.conv1.weight']
    dest.module.conv1.weight.data = tmp
    tmp = src['state_dict']['module.bn1.weight']
    dest.module.bn1.weight.data = tmp
    tmp = src['state_dict']['module.bn1.bias']
    dest.module.bn1.bias.data = tmp
    tmp = src['state_dict']['module.bn1.running_mean']
    dest.module.bn1.running_mean.data = tmp
    tmp = src['state_dict']['module.bn1.running_var']
    dest.module.bn1.running_var.data = tmp
    tmp = src['state_dict']['module.bn1.num_batches_tracked']
    dest.module.bn1.num_batches_tracked.data = tmp
    tmp = src['state_dict']['module.fc.weight']
    dest.module.fc.weight.data = tmp
    tmp = src['state_dict']['module.fc.bias']
    dest.module.fc.bias.data = tmp
    dest.module.group2_ds0._modules['0'].weight.data = src['state_dict']['module.layer2.0.downsample.0.weight']
    dest.module.group2_ds0._modules['1'].weight.data = src['state_dict']['module.layer2.0.downsample.1.weight']
    dest.module.group2_ds0._modules['1'].bias.data = src['state_dict']['module.layer2.0.downsample.1.bias']
    dest.module.group2_ds0._modules['1'].running_mean.data = src['state_dict']['module.layer2.0.downsample.1.running_mean']
    dest.module.group2_ds0._modules['1'].running_var.data = src['state_dict']['module.layer2.0.downsample.1.running_var']
    dest.module.group2_ds0._modules['1'].num_batches_tracked.data = src['state_dict']['module.layer2.0.downsample.1.num_batches_tracked']

    dest.module.group3_ds0._modules['0'].weight.data = src['state_dict']['module.layer3.0.downsample.0.weight']
    dest.module.group3_ds0._modules['1'].weight.data = src['state_dict']['module.layer3.0.downsample.1.weight']
    dest.module.group3_ds0._modules['1'].bias.data = src['state_dict']['module.layer3.0.downsample.1.bias']
    dest.module.group3_ds0._modules['1'].running_mean.data = src['state_dict']['module.layer3.0.downsample.1.running_mean']
    dest.module.group3_ds0._modules['1'].running_var.data = src['state_dict']['module.layer3.0.downsample.1.running_var']
    dest.module.group3_ds0._modules['1'].num_batches_tracked.data = src['state_dict']['module.layer3.0.downsample.1.num_batches_tracked']

    for i in [1, 2, 3]:
        for j in range(12):
            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.conv1.weight']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).conv1.weight.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn1.weight']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn1.weight.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn1.bias']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn1.bias.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn1.running_mean']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn1.running_mean.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn1.running_var']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn1.running_var.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn1.num_batches_tracked']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn1.num_batches_tracked.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.conv2.weight']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).conv2.weight.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn2.weight']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn2.weight.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn2.bias']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn2.bias.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn2.running_mean']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn2.running_mean.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn2.running_var']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn2.running_var.data = tmp

            tmp = src['state_dict']['module.layer' + str(i) + '.' + str(j) + '.bn2.num_batches_tracked']
            getattr(dest.module, 'group' + str(i) + '_layer' + str(j)).bn2.num_batches_tracked.data = tmp


def run_training(args):
    # create model
    model = cifar10_rnn_gate_74(**args.signsgd_config)
    # for m in model.parameters():
    #     m.requires_grad = False
    # for m in model.fc.parameters():
    #     m.requires_grad = True
    # for g in [1, 2, 3]:
    #     for i in range(12):
    #         tmp = getattr(model, 'group{}_gate{}'.format(g, i))
    model.install_gate()
    model = torch.nn.DataParallel(model).cuda()


    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume)
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))

            args.start_iter = 0
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            # translate(model, checkpoint)
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
    # model.module.install_gate()
    # model = model.cuda()

    cudnn.benchmark = True

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = CusSGD(filter(lambda p: p.requires_grad,
    #                                    model.parameters()),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cp_energy_record = AverageMeter()
    skip_ratios = ListAverageMeter()

    end = time.time()
    for i in range(0, args.iters):

        rand_flag = random.uniform(0, 1) > 0.5
        # if tmp_rand_flag > 0.5:
        #     for m in model.module.parameters():
        #         m.requires_grad = True
        #     tmp_count = 0
        #     tmp_rand = random.randint(1, 36)
        #     for g in [1, 2, 3]:
        #         for j in range(12):
        #             tmp_count += 1
        #             if tmp_count > tmp_rand:
        #                 break
        #             tmp = getattr(model.module, 'group{}_gate{}'.format(g, j))
        #             for m in tmp.parameters():
        #                 m.requires_grad = False
        #             tmp = getattr(model.module, 'group{}_layer{}'.format(g, j))
        #             for m in tmp.parameters():
        #                 m.requires_grad = False
        #             tmp = getattr(model.module, 'group{}_ds{}'.format(g, j))
        #             if tmp is not None:
        #                 for m in tmp.parameters():
        #                     m.requires_grad = False
        #         if tmp_count >= tmp_rand:
        #             break
        # else:
        #     for m in model.parameters():
        #             m.requires_grad = True

        # if tmp_rand_flag > 0.5:
        #     for m in model.module.parameters():
        #         m.requires_grad = False
        #     for m in model.module.fc.parameters():
        #         m.requires_grad = True
        #     # for j in range(12):
        #     #     tmp = getattr(model.module, 'group{}_gate{}'.format(3, j))
        #     #     for m in tmp.parameters():
        #     #         m.requires_grad = True
        #     #     tmp = getattr(model.module, 'group{}_layer{}'.format(3, j))
        #     #     for m in tmp.parameters():
        #     #         m.requires_grad = True
        #     #     tmp = getattr(model.module, 'group{}_ds{}'.format(3, j))
        #     #     if tmp is not None:
        #     #         for m in tmp.parameters():
        #     #             m.requires_grad = True
        # else:
        #     for m in model.parameters():
        #         m.requires_grad = True
        # for m in model.parameters():
        #     m.requires_grad = True
        # tmp_mode = 4
        # if 0 < tmp_rand_flag <= 0.25:
        #     tmp_mode = 1
        # elif 0.25 < tmp_rand_flag <= 0.5:
        #     tmp_mode = 2
        # elif 0.5 < tmp_rand_flag <= 0.75:
        #     tmp_mode = 3

        # for g in [1, 2, 3]:
        #     if g >= tmp_mode:
        #         break
        #     for j in range(12):
        #         tmp = getattr(model.module, 'group{}_gate{}'.format(g, j))
        #         for m in tmp.parameters():
        #             m.requires_grad = False
        #         tmp = getattr(model.module, 'group{}_layer{}'.format(g, j))
        #         for m in tmp.parameters():
        #             m.requires_grad = False
        #         tmp = getattr(model.module, 'group{}_ds{}'.format(g, j))
        #         if tmp is not None:
        #             for m in tmp.parameters():
        #                 m.requires_grad = False


        model.train()
        adjust_learning_rate(args, optimizer, i)

        input, target = next(iter(train_loader))

        # measuring data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = Variable(input, requires_grad=True).cuda()
        target_var = Variable(target).cuda()

        # compute output
        if rand_flag:
            optimizer.zero_grad()
            optimizer.step()
            global skip_count
            skip_count += 1
            continue

        output, masks, logprobs = model(input_var)

        energy_parameter = np.ones(35,)
        energy_parameter /= energy_parameter.max()

        energy_cost = 0
        energy_all = 0
        for layer in range(len(energy_parameter)):
            energy_cost += masks[layer].sum() * energy_parameter[layer]
            energy_all += reduce((lambda x, y: x * y), masks[layer].shape) * energy_parameter[layer]

        cp_energy = (energy_cost.item() / energy_all.item()) * 100
        global training_cost
        training_cost += (cp_energy / 100) * 0.51 * args.batch_size
        # if tmp_mode == 1:
        #     training_cost += (cp_energy / 100) * 0.51 * args.batch_size
        # elif tmp_mode == 2:
        #     training_cost += (cp_energy / 100) * (0.17 * args.batch_size + (2 / 3) * 0.34 * args.batch_size)
        # elif tmp_mode == 3:
        #     training_cost += (cp_energy / 100) * (0.17 * args.batch_size + (1 / 3) * 0.34 * args.batch_size)
        # else:
        #     training_cost += (cp_energy / 100) * 0.17 * args.batch_size

        # if tmp_rand_flag > 0.5:
        #     # training_cost += (cp_energy / 100) * 0.17 * args.batch_size + float((36 - tmp_rand) / 36) * 0.34 * args.batch_size
        #     training_cost += (cp_energy / 100) * 0.17 * args.batch_size + 0
        # else:
        #     training_cost += (cp_energy / 100) * 0.51 * args.batch_size
        energy_cost *= args.beta
        if cp_energy <= args.minimum:
            reg = -1
        else:
            reg = 1
        if args.energy:
            loss = criterion(output, target_var) + energy_cost * reg
        else:
            loss = criterion(output, target_var)

        # collect skip ratio of each layer
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        # loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        writer.add_scalar('data/train_error', 100 - prec1, i-skip_count)
        writer.add_scalar('data/train_comp_using', cp_energy, i-skip_count)
        writer.add_scalar('data/train_cost_Gops', training_cost, i-skip_count)
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        cp_energy_record.update(cp_energy, 1)
        skip_ratios.update(skips, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # repackage hidden units for RNN Gate
        if args.gate_type == 'rnn':
            model.module.control.repackage_hidden()

        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
            logging.info("Iter: [{0}/{1}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                         'Energy_ratio: {cp_energy_record.val:.3f}({cp_energy_record.avg:.3f})\t'.format(
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1,
                            cp_energy_record=cp_energy_record)
            )
            # for idx in range(skip_ratios.len):
            #     logging.info(
            #         "{} layer skipping = {:.3f}({:.3f})".format(
            #             idx,
            #             skip_ratios.val[idx],
            #             skip_ratios.avg[idx],
            #         )
            #     )

        # evaluate every 1000 steps
        if (i % args.eval_every == 0 and i > 0) or (i == (args.iters-1)):
            prec1 = validate(args, test_loader, model, criterion)
            writer.add_scalar('data/test_error', 100 - prec1, i-skip_count)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                           'checkpoint_{:05d}.pth.tar'.format(
                                               i))
            save_checkpoint({
                'iter': i,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
                is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                          'checkpoint_latest'
                                                          '.pth.tar'))


def validate(args, test_loader, model, criterion):
    # for i in range(128):
    #     scipy.misc.imsave('/home/yw68/skipnet/cifar/images_label/0{}.png'.format(i),
    #                       test_loader_v.dataset.data[i])

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    cp_energy_record = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        if i == len(test_loader) - 1:
            break
        target = target.cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()
        # compute output
        output, masks, logprobs = model(input_var)

        energy_parameter = np.ones(35, )
        # for index, item in enumerate(energy_parameter):
        #     if index <= 10:
        #         energy_parameter[index] /= 16
        #     elif index <= 22:
        #         energy_parameter[index] /= 32
        #     else:
        #         energy_parameter[index] /= 64

        energy_parameter /= energy_parameter.max()

        energy_cost = 0
        energy_all = 0
        for layer in range(len(energy_parameter)):
            energy_cost += masks[layer].sum() * energy_parameter[layer]
            energy_all += reduce((lambda x, y: x * y), masks[layer].shape) * energy_parameter[layer]
        cp_energy = (energy_cost.item() / energy_all.item()) * 100

        skips = [mask.data.le(0.5).float().mean().item() for mask in masks]

        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))
        losses.update(loss.data.item(), input.size(0))
        batch_time.update(time.time() - end)
        cp_energy_record.update(cp_energy, 1)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'
                'Energy_ratio: {cp_energy_record.val:.3f}({cp_energy_record.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    cp_energy_record=cp_energy_record,
                )
            )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        # logging.info(
        #     "{} layer skipping = {:.3f}".format(
        #         idx,
        #         skip_ratios.avg[idx],
        #     )
        # )
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model.install_gate()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # translate(model, checkpoint)
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
    # model.module.install_gate()
    # model = model.cuda()
    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    test_loader_v = prepare_test_data_v(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, test_loader, model, criterion, test_loader_v)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best_eic.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """divide lr by 10 at 32k and 48k """
    if args.warm_up and (_iter < 400):
        lr = 0.01
    elif int(32000 * 4/3) <= _iter < int(48000 * 4/3):
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= int(48000 * 4/3):
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    # if _iter % args.eval_every == 0:
    #     logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
