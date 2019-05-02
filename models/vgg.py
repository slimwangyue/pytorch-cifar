'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from numba import vectorize, cuda
import numpy as np
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                tmp_conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)

                def hookFunc(conv2d_obj, saved_variables, inputs):
                    x_grad = None
                    w_grad = None
                    bias_grad = None

                    x_in = saved_variables[0]
                    w = conv2d_obj.weight
                    b = conv2d_obj.bias
                    grads = inputs[0]

                    if x_in is not None and grads is not None:
                        x_in_low = torch.from_numpy(quantize_weights_waste(x_in.cpu().numpy(), 32)).cuda()
                        w_low = torch.from_numpy(quantize_weights_waste(w.cpu().numpy(), 32)).cuda()
                        grads_low = torch.from_numpy(quantize_weights_waste(grads.cpu().numpy(), 32)).cuda()
                    else:
                        x_in_low = None
                        w_low = None
                        grads_low = None

                    if x_in is not None and w is not None:
                        x_grad = torch.nn.grad.conv2d_input(x_in_low.shape, w_low, grads_low,
                                                            stride=conv2d_obj.stride, padding=conv2d_obj.padding)
                        # x_grad = torch.from_numpy(quantize_weights_waste(x_grad.astype('float32'), 8)).cuda()

                    if x_in is not None and w is not None:
                        w_grad = torch.nn.grad.conv2d_weight(x_in_low, w_low.shape, grads_low,
                                                             stride=conv2d_obj.stride, padding=conv2d_obj.padding)
                        # w_grad = (w_grad > 0).float() * 2 - 1
                        # w_grad = torch.from_numpy(quantize_weights_waste(w_grad.astype('float32'), 8)).cuda()

                    if b is not None:
                        bias_grad = torch.ones(b.shape, device=torch.device('cuda:0')) * torch.sum(grads_low,
                                                                                                   dim=(0, 2, 3))

                    return x_grad, w_grad, bias_grad

                tmp_conv2d.register_backward_hook(hookFunc)

                layers += [tmp_conv2d,
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()


@vectorize('float32(float32, int8)', target='cuda')
def numba_quantize(feature_map, bit_precision):
    flag = feature_map == 1
    delta = 1 / pow(2, bit_precision - 1)
    flag *= delta
    return (np.int32((feature_map + 1) / delta) * delta) - 1 - flag


# def quantize_weights_waste(feature_map, precision, norm_list=None, norm_list_true=None):
#     return feature_map.cpu().detach().numpy()

def quantize_weights_waste(feature_map, precision, norm_list=None, norm_list_true=None):
    if precision == 32:
        return feature_map
    shape = feature_map.shape
    # max_num = feature_map.cpu().detach().numpy().abs().max()
    max_num = np.abs(feature_map).max()
    if norm_list_true is not None:
       norm_list_true.append(max_num)
    norm = nearestpow2(max_num)
    if norm_list is not None:
       norm_list.append(norm)
    input_quan = feature_map
    input_quan = input_quan.astype(np.float32)
    fm_quan = numba_quantize((input_quan / norm).astype(np.float32), precision) * norm
    # tmp = torch.from_numpy(np.array(fm_quan).reshape(shape).astype('float32')).cuda()
    return np.array(fm_quan).reshape(shape).astype('float32')


def nearestpow2(x):
    flag_nag = False
    if x == 0:
        return 0
    if x < 0:
        x = -x
        flag_nag = True
    tmp_ceil = 2**math.ceil(math.log2(x))
    tmp_floor = 2**math.floor(math.log2(x))
    if abs(tmp_ceil - x) > abs(tmp_floor - x):
        if flag_nag:
            return -tmp_floor
        else:
            return tmp_floor
    else:
        if flag_nag:
            return -tmp_ceil
        else:
            return tmp_ceil
