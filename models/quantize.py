from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['max_values', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,
                      reduce_type='mean', keepdim=False):
    with torch.no_grad():
        x_flat_abs = x.abs().flatten(*flatten_dims)
        if x_flat_abs.dim() == 1:
            max_values = _deflatten_as(x_flat_abs.max(), x)
        else:
            max_values = _deflatten_as(x_flat_abs.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        # max_values = max_values[0].detach().cpu().item()
        return QParams(max_values=max_values, num_bits=num_bits)


class FPQuantizeFunction(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=32, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        # the following two `if` statements are just used to save computation
        if qparams is not None and qparams.num_bits >= 32:
            return input
        if qparams is None:
            assert num_bits is not None
            if num_bits >= 32:
                return input

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            qparams = calculate_qparams(
                output_abs, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        num_bits = qparams.num_bits
        max_values = qparams.max_values
        min_values = - max_values if signed else 0.
        delta = (max_values - min_values) / 2.**num_bits
        qmin, qmax = 0.0, 2**num_bits - 1
        with torch.no_grad():
            # output.clamp_(min_values, max_values)
            # max_locs = (output == max_values).float()
            # output.sub_(min_values).div_(delta).round_().sub_(max_locs)
            output.sub_(min_values).div_(delta).clamp_(qmin,qmax).round_()

            if dequantize:
                output.mul_(delta).add_(min_values)

            # output_abs.add_(qmin * scale - zero_point).div_(scale)
            # if stochastic:
            #     noise = output_abs.new(output.shape).uniform_(-0.5, 0.5)
            #     output_abs.add_(noise)
            # # quantize
            # output_abs.clamp_(qmin, qmax).round_()

            # if dequantize:
            #     output.mul_(scale).add_(
            #         zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class FPQuantizeGradFunction(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=32, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=True, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.dequantize = dequantize
        ctx.signed = signed
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.qparams is not None and ctx.qparams.num_bits >= 32:
            return grad_output

        if ctx.qparams is None:
            assert ctx.num_bits is not None
            if ctx.num_bits >= 32:
                return grad_output

        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits,
                    flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                    reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None,
             flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,
             dequantize=True, signed=False, stochastic=False, inplace=False):
    return FPQuantizeFunction().apply(
        x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None,
                  flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0,
                  dequantize=True, signed=True, stochastic=False):
    return FPQuantizeGradFunction().apply(
        x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class Quantize(nn.Module):
    """docstring for Quantize"""

    def __init__(self, num_bits=8, num_bits_grad=32, shape_measure=(1,),
                 dequantize=True, signed=False, stochastic=False, momentum=0.1):
        super(Quantize, self).__init__()
        self.register_buffer('running_max_values', torch.zeros(*shape_measure))
        # self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.momentum = momentum
        self.dequantize = dequantize
        self.signed = signed
        self.stochastic = stochastic
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad

    def forward(self, input, qparams=None):

        # quantize input
        if self.num_bits < 32:
            if self.training:
                if qparams is None:
                    qparams = calculate_qparams(
                        input, num_bits=self.num_bits, flatten_dims=(1,-1), reduce_dim=0)
                with torch.no_grad():
                    momentum = self.momentum
                    self.running_max_values.mul_(momentum).add_(
                        qparams.max_values * (1 - momentum))
            else:
                qparams = QParams(max_values=self.running_max_values,
                                  num_bits=self.num_bits)
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               signed=self.signed, stochastic=self.stochastic, inplace=False)
        else:
            q_input = input

        # quantize grad
        if self.training and self.num_bits_grad < 32:
            q_input = quantize_grad(
                q_input, num_bits=self.num_bits_grad, flatten_dims=(1, -1))

        return q_input


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(
                    output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, (B * H * W) // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = (mean_max - mean_min) * scale_fix
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean * (1 - self.momentum))

                self.running_var.mul_(self.momentum).add_(
                    scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        # scale = quantize(scale, num_bits=self.num_bits, min_value=float(
        #     scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, -1, 1, 1)) / \
            (scale.view(1, -1, 1, 1) + self.eps)

        if self.weight is not None:
            qweight = self.weight
            # qweight = quantize(self.weight, num_bits=self.num_bits,
            #                    min_value=float(self.weight.min()),
            #                    max_value=float(self.weight.max()))
            out = out * qweight.view(1, -1, 1, 1)

        if self.bias is not None:
            qbias = self.bias
            # qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, -1, 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(
                out, num_bits=self.num_bits_grad, flatten_dims=(1, -1))

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class RangeBN1d(RangeBN):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN1d, self).__init__(num_features, dim, momentum,
                                        affine, num_chunks, eps, num_bits, num_bits_grad)
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1), flatten_dims=(1, -1))

if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
