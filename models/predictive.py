"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch.nn.modules.utils import _pair
from torch.autograd import Function

from models.quantize import calculate_qparams, quantize, quantize_grad


# Inherit from Function
class PredictiveForwardMixingFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, q_out, msb_out, msb_bits_grad):
        ctx.save_for_backward(msb_out)
        ctx.msb_bits_grad = msb_bits_grad

        if msb_out is None:
            return q_out.clone()
        else:
            with torch.no_grad():
                nne_locs = (msb_out >= 0).detach().float()
                return q_out * nne_locs

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        msb_out = ctx.saved_tensors[0]
        msb_bits_grad = ctx.msb_bits_grad
        grad_q_out = grad_msb_out = None

        with torch.no_grad():
            grad_q_out = grad_output.clone()
            if msb_out is not None:
                grad_msb_out = quantize(
                    grad_output.clone(), num_bits=msb_bits_grad,
                    flatten_dims=(1,-1), signed=True)

            return grad_q_out, grad_msb_out, None


class PredictiveWeightQuantFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, weight, num_bits_weight, msb_bits_weight,
                threshold, sparsify, sign):
        ctx.threshold = threshold
        ctx.sparsify = sparsify
        ctx.sign = sign

        with torch.no_grad():
            # q_weight
            if num_bits_weight is not None and num_bits_weight < 32:
                q_weight = quantize(
                    weight, num_bits=num_bits_weight,
                    flatten_dims=(1,-1), reduce_dim=None, signed=True)
            else:
                q_weight = weight

            # msb_weight
            if msb_bits_weight is None:
                return (q_weight,)
            elif msb_bits_weight < 32:
                msb_weight = quantize(
                    weight, num_bits=msb_bits_weight,
                    flatten_dims=(1,-1), reduce_dim=None, signed=True)
            else:
                msb_weight = weight.clone()

            return q_weight, msb_weight

    # This function has two outputs, so it gets two gradients
    @staticmethod
    def backward(ctx, *grad_output):
        grad_weight = None
        grad_q_weight = grad_output[0]
        grad_msb_weight = grad_output[1] if len(grad_output) > 1 else None

        with torch.no_grad():
            if grad_msb_weight is not None:
                large_locs = (grad_msb_weight.abs() >= ctx.threshold).detach().float()
                if ctx.sparsify:
                    grad_weight = large_locs * grad_msb_weight
                else:
                    grad_weight = large_locs * grad_msb_weight + (1 - large_locs) * grad_q_weight
            else:
                grad_weight = grad_q_weight.clone()

            if ctx.sign:
                grad_weight.sign_()

            return grad_weight, None, None, None, None, None


class PredictiveBiasQuantFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, bias, num_bits_bias, msb_bits_bias, threshold, sparsify, sign):
        ctx.threshold = threshold
        ctx.sparsify = sparsify
        ctx.sign = sign

        with torch.no_grad():
            # q_weight
            if num_bits_bias is not None and num_bits_bias < 32:
                q_bias = quantize(bias, num_bits=num_bits_bias,
                                  flatten_dims=(0,-1), reduce_dim=0, signed=True)
            else:
                q_bias = bias

            # msb_weight
            if msb_bits_bias is None:
                return (q_bias,)
            elif msb_bits_bias < 32:
                msb_bias = quantize(bias, num_bits=msb_bits_bias,
                                    flatten_dims=(0,-1), reduce_dim=0, signed=True)
            else:
                msb_bias = bias.clone()

            return q_bias, msb_bias

    # This function has two outputs, so it gets two gradients
    @staticmethod
    def backward(ctx, *grad_output):
        grad_bias = None
        grad_q_bias = grad_output[0]
        grad_msb_bias = grad_output[1] if len(grad_output) > 1 else None

        with torch.no_grad():
            if grad_msb_bias is not None:
                large_locs = (grad_msb_bias.abs() >= ctx.threshold).detach().float()
                if ctx.sparsify:
                    grad_bias = large_locs * grad_msb_bias
                else:
                    grad_bias = large_locs * grad_msb_bias + (1 - large_locs) * grad_q_bias
            else:
                grad_bias = grad_q_bias.clone()

            if ctx.sign:
                grad_bias.sign_()

            return grad_bias, None, None, None, None, None


def mixing_output(q_out, msb_out, msb_bits_grad=16):
    return PredictiveForwardMixingFunction.apply(
        q_out, msb_out, msb_bits_grad)


def quant_weight(weight, num_bits_weight=8, msb_bits_weight=4,
                 threshold=5e-4, sparsify=False, sign=False):
    return PredictiveWeightQuantFunction.apply(
        weight, num_bits_weight, msb_bits_weight, threshold, sparsify, sign)


def quant_bias(weight, num_bits_bias=16, msb_bits_bias=8,
               threshold=5e-4, sparsify=False, sign=False):
    return PredictiveBiasQuantFunction.apply(
        weight, num_bits_bias, msb_bits_bias, threshold, sparsify, sign)
