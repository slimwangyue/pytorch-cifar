"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function

from models.quantize import calculate_qparams, quantize, quantize_grad
from models.predictive import mixing_output, quant_weight


# Inherit from Function
class PredictiveConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, # NOTE: CONV layer has no bias by default # the above two lines are the same as Conv2d
                 num_bits_weight=8, # num_bits_bias=16, # how the weight and the bias will be quantized
                 input_signed=False, # whether the input is signed or unsigned
                 predictive_forward=True, predictive_backward=True,
                 msb_bits=4, msb_bits_weight=4, msb_bits_grad=16,
                 threshold=5e-5, sparsify=False, sign=False, writer=None): # used in `backward()`
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PredictiveConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)

        self.num_bits_weight = num_bits_weight
        self.input_signed = input_signed
        self.predictive_forward = predictive_forward
        self.predictive_backward = predictive_backward
        self.msb_bits = msb_bits
        self.msb_bits_weight = msb_bits_weight
        self.msb_bits_grad = msb_bits_grad
        self.threshold = threshold
        self.sparsify = sparsify
        self.sign = sign

        if not self.predictive_backward:
            self.msb_bits_grad = None
            if not self.predictive_forward:
                self.msb_bits = self.msb_bits_weight = None
            else:
                assert self.msb_bits is not None and self.msb_bits_weight is not None

    def forward(self, input):
        # See the autograd section for explanation of what happens here.

        # Quantize `input` to get `msb_input`
        if self.msb_bits is not None:
            msb_input_qparams = calculate_qparams(input.detach(), num_bits=self.msb_bits,
                                                  flatten_dims=(1,-1), reduce_dim=0)
            msb_input = quantize(input, qparams=msb_input_qparams, signed=self.input_signed)
        else:
            msb_input = None

        # Quantize weight
        weights = quant_weight(
            self.weight, num_bits_weight=self.num_bits_weight,
            msb_bits_weight=self.msb_bits_weight, threshold=self.threshold,
            sparsify=self.sparsify, sign=self.sign)
        q_weight = weights[0]
        msb_weight = weights[1] if len(weights) > 1 else None

        # No bias for CONV layers
        q_bias = None

        # Q-branch
        q_output = F.conv2d(input, q_weight, bias=q_bias, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)
        # MSB-branch
        if self.predictive_forward and msb_input is not None and msb_weight is not None:
            msb_output = F.conv2d(msb_input, msb_weight, bias=q_bias, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            msb_output = None

        # Mixing `q_output` and `msb_output`
        output = mixing_output(q_output, msb_output, self.msb_bits_grad)

        return output
