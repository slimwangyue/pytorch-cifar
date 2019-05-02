'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from numba import vectorize, cuda
import numpy as np


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

                def hookFunc(conv2d_obj, saved_variables, grads):
                    x_grad = None
                    w_grad = None
                    bias_grad = None
                    x_in, w, b = saved_variables

                    x_in_low = predict(x_in)
                    w_low = predict(w)
                    grads_low = predict(grads[0])

                    if x_in is not None and w is not None:
                        x_grad = torch.nn.grad.conv2d_input(x_in_low.shape, w_low, grads_low, stride=conv2d_obj.stride, padding=conv2d_obj.padding)
                        x_grad = x_grad

                    if x_in is not None and w is not None:
                        w_grad = torch.nn.grad.conv2d_weight(x_in_low, w_low.shape, grads_low, stride=conv2d_obj.stride, padding=conv2d_obj.padding)
                        w_grad = (w_grad > 0).float() * 2 - 1
                        w_grad = w_grad

                    if b is not None:
                        bias_grad = torch.ones(b.shape, device=torch.device('cuda:0')) * torch.sum(grads_low, dim=(0, 2, 3))

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


def predict(tensor):
    return tensor
