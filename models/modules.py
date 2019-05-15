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


# Inherit from Function
class PredictiveSignConv2dFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias,
                stride, padding, dilation, groups,
                mbs_bits, mbs_bits_weight, mbs_bits_grad):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        with torch.no_grad():
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = bias.new_ones(bias.shape) * torch.sum(grad_output, dim=(0, 2, 3))

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None)


class PredictiveSignConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_bits_weight=8, num_bits_bias=8,
                 mbs_bits=4, mbs_bits_weight=4, mbs_bits_grad=16):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PredictiveSignConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

        self.num_bits_weight = num_bits_weight
        self.num_bits_bias = num_bits_bias
        self.mbs_bits = mbs_bits
        self.mbs_bits_weight = mbs_bits_weight
        self.mbs_bits_grad = mbs_bits_grad

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)

        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits_bias, flatten_dims=(0, -1))
        else:
            qbias = None

        return PredictiveSignConv2dFunction.apply(
            input, qweight, qbias, self.stride,
            self.padding, self.dilation, self.groups,
            self.mbs_bits, self.mbs_bits_weight, self.mbs_bits_grad)


