'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import math

from models.new_conv import PredictiveSignConv2d
from models.new_linear import PredictiveSignLinear
from models.quantize import Quantize

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name,
                 num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                 msb_bits, msb_bits_grad, msb_bits_weight, threshold,
                 writer=None):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        # self.writer = writer
        self.counter = 0

    def forward(self, x):
        out = self.features(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg,
                     num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                     msb_bits, msb_bits_grad, msb_bits_weight, threshold):
        layers = []
        # Input quantization. Inputs are signed after normalization.
        input_quant = Quantize(
            num_bits=num_bits, num_bits_grad=num_bits_grad, shape_measure=(1,1,1,1,),
            dequantize=True, signed=True, stochastic=False, momentum=0.1)
        layers.append(input_quant)
        in_channels = 3
        for i, x in enumerate(cfg):
            input_signed = True if i == 0 else False
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = PredictiveSignConv2d(
                    in_channels, x, kernel_size=3, padding=1, bias=False,
                    num_bits_weight=num_bits_weight, num_bits_bias=num_bits_bias,
                    input_signed=input_signed,
                    msb_bits=msb_bits, msb_bits_weight=msb_bits_weight,
                    msb_bits_grad=msb_bits_grad, threshold=threshold)

                # Activations after ReLU layers are unsigned
                act_quant = Quantize(
                    num_bits=num_bits, num_bits_grad=num_bits_grad, shape_measure=(1,1,1,1,),
                    dequantize=True, signed=False, stochastic=False, momentum=0.1)

                layers += [conv2d,
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           act_quant]

                in_channels = x

        act_quant = Quantize(
            num_bits=num_bits, num_bits_grad=num_bits_grad, shape_measure=(1,1,1,1,),
            dequantize=True, signed=False, stochastic=False, momentum=0.1)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1),
                   act_quant]

        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
