import os
import torch
import random
import numpy as np
from functools import reduce
import time


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, args, writer, total_i, skip_count, training_cost):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    # cur = time.time()
    for i, (input, target) in enumerate(loader):
        # print('loading time', time.time() - cur)
        rand_flag = random.uniform(0, 1) > 0.5
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if rand_flag:
            optimizer.zero_grad()
            optimizer.step()
            skip_count += 1
            continue
        # cur = time.time()
        output = model(input_var)
        # print('comp time', time.time() - cur)
        # energy_parameter = np.ones(53, )
        # energy_parameter /= energy_parameter.max()
        #
        # energy_cost = 0
        # energy_all = 0
        # for layer in range(len(energy_parameter)):
        #     energy_cost += masks[layer].sum() * energy_parameter[layer]
        #     energy_all += reduce((lambda x, y: x * y), masks[layer].shape) * energy_parameter[layer]
        #
        # cp_energy = (energy_cost.item() / energy_all.item()) * 100
        # training_cost += (cp_energy / 100) * 0.51 * args.batch_size
        #
        # energy_cost *= args.beta
        # if cp_energy <= args.minimum:
        #     reg = -1
        # else:
        #     reg = 1
        # if args.energy:
        #     loss = criterion(output, target_var) + energy_cost * reg
        # else:
        #     loss = criterion(output, target_var)
        #
        # # collect skip ratio of each layer
        # skips = [mask.data.le(0.5).float().mean() for mask in masks]

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        writer.add_scalar('data/train_error1', 100 - prec1, total_i)
        writer.add_scalar('data/train_error5', 100 - prec5, total_i)
        # writer.add_scalar('data/train_comp_using', cp_energy, total_i)
        # writer.add_scalar('data/train_cost_Gops', training_cost, total_i)

        # model.module.control.repackage_hidden()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
        total_i += 1

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }, total_i, training_cost


def eval(loader, model, criterion, writer, total_i, skip_count):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

        energy_parameter = np.ones(53, )

        energy_parameter /= energy_parameter.max()

        energy_cost = 0
        energy_all = 0
        # for layer in range(len(energy_parameter)):
        #     energy_cost += masks[layer].sum() * energy_parameter[layer]
        #     energy_all += reduce((lambda x, y: x * y), masks[layer].shape) * energy_parameter[layer]
        # cp_energy = (energy_cost.item() / energy_all.item()) * 100
        #
        # skips = [mask.data.le(0.5).float().mean().item() for mask in masks]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        writer.add_scalar('data/test_error1', 100 - prec1, total_i)
        writer.add_scalar('data/test_error5', 100 - prec5, total_i)
        # writer.add_scalar('data/test_comp_using', cp_energy, total_i)




    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


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
