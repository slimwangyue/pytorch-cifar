import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import scipy.misc
from models.new_conv import PredictiveSignConv2d
from models.new_linear import PredictiveSignLinear
from models.quantize import Quantize


def conv3x3(in_planes, out_planes, num_bits_weight, num_bits_bias,
            msb_bits, msb_bits_grad, msb_bits_weight, threshold, stride=1, input_signed=False):
    "3x3 convolution with padding"
    return PredictiveSignConv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False, stride=stride,
                                num_bits_weight=num_bits_weight, num_bits_bias=num_bits_bias,
                                input_signed=input_signed, msb_bits=msb_bits, msb_bits_weight=msb_bits_weight,
                                msb_bits_grad=msb_bits_grad, threshold=threshold)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_bits, num_bits_weight, num_bits_bias, num_bits_grad, msb_bits,
                 msb_bits_grad, msb_bits_weight, threshold, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, num_bits_weight, num_bits_bias, msb_bits,
                             msb_bits_grad, msb_bits_weight, threshold, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, num_bits_weight, num_bits_bias, msb_bits,
                             msb_bits_grad, msb_bits_weight, threshold)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.act_quant = Quantize(num_bits=num_bits, num_bits_grad=num_bits_grad, shape_measure=(1, 1, 1, 1,),
                                  dequantize=True, signed=False, stochastic=False, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.act_quant(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.act_quant(out)
        return out

########################################
# SkipNet+SP with Recurrent Gate       #
########################################


# For Recurrent Gate
def repackage_hidden(h):
    """ to reduce memory usage"""
    if h is None:
        return None
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """Recurrent Gate definition.
    Input is already passed through average pooling and embedding."""
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm', output_channel=1):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, output_channel)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda(), requires_grad=True),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda(), requires_grad=True))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        # prob = nn.functional.relu(prob - 0.1)

        tmp = torch.rand_like(prob)
        disc_prob = (prob > tmp).float().detach() - \
                    prob.detach() + prob

        disc_prob = disc_prob.view(batch_size, -1, 1, 1)
        return disc_prob, prob
    #
    # def forward(self, x, jump):
    #     # Take the convolution output of each step
    #     batch_size = x.size(0)
    #     self.rnn.flatten_parameters()
    #     out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)
    #
    #     proj = self.proj(out.squeeze())
    #     prob = self.prob(proj)
    #     if jump != -1:
    #         prob = torch.nn.functional.avg_pool1d(prob.view(batch_size, 1, -1), kernel_size=jump, stride=jump, padding=0)
    #     prob.squeeze()
    #
    #     disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
    #
    #     disc_prob = disc_prob.view(batch_size, -1, 1, 1)
    #     return disc_prob, prob


class ResNetRecurrentGateSP(nn.Module):
    """SkipNet with Recurrent Gate Model"""
    def __init__(self, block, layers,
                 num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                 msb_bits, msb_bits_grad, msb_bits_weight, threshold,
                 writer=None,
                 num_classes=10, embed_dim=10,
                 hidden_dim=10, gate_type='rnn'):
        self.inplanes = 16
        super(ResNetRecurrentGateSP, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16, num_bits_weight, num_bits_bias, msb_bits, msb_bits_grad, msb_bits_weight,
                             threshold, stride=1, input_signed=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self._make_group(block, 16, layers[0], num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                         msb_bits, msb_bits_grad, msb_bits_weight, threshold, group_id=1, pool_size=32)
        self._make_group(block, 32, layers[1], num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                         msb_bits, msb_bits_grad, msb_bits_weight, threshold, group_id=2, pool_size=16)
        self._make_group(block, 64, layers[2], num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                         msb_bits, msb_bits_grad, msb_bits_weight, threshold, group_id=3, pool_size=8)

        # define recurrent gating module
        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = PredictiveSignLinear(512, 10, num_bits_weight=num_bits_weight, num_bits_bias=num_bits_bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def install_gate(self):
        self.control = RNNGate(self.embed_dim, self.hidden_dim, rnn_type='lstm', output_channel=1)

    def _make_group(self, block, planes, layers, num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                    msb_bits, msb_bits_grad, msb_bits_weight, threshold, group_id=1, pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                                       msb_bits, msb_bits_grad, msb_bits_weight, threshold, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                       msb_bits, msb_bits_grad, msb_bits_weight, threshold,
                       stride=1, pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )

        layer = block(self.inplanes, planes, stride, num_bits, num_bits_weight, num_bits_bias, num_bits_grad,
                      msb_bits, msb_bits_grad, msb_bits_weight, threshold, downsample)

        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))
        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def forward(self, x):

        img_list = []


        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)


        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer

        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)

                # if g == 0 and i == 6:
                #     for j in range(16):
                #         img_list.append(x[99][j].cpu().detach().numpy())
                #         new_img = img_list[j]
                #         new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
                #         scipy.misc.imsave('/home/yw68/skipnet/cifar/images_fm/{}_no_test.png'.format(j), new_img)

                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                # new mask is taking the current output
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev

                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                # control = getattr(self, 'control{}'.format(min(3, g + 1 + (i == self.num_layers[g] - 1))))
                mask, gprob = self.control(gate_feature)
                # if i == self.num_layers[g] - 1 and g != 2:
                #     mask, grob = self.control(gate_feature, int(64 / (2**(g+5))))
                # else:
                #     mask, grob = self.control(gate_feature, int(64 / (2**(g+4))))
                gprobs.append(gprob)
                masks.append(mask.squeeze())

        # last block doesn't have gate module
        del masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs


# For CIFAR-10
def cifar10_rnn_gate_38(pretrained=False, **kwargs):
    """SkipNet-38 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [6, 6, 6], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_74(num_bits, num_bits_weight, num_bits_bias, num_bits_grad, msb_bits,
                        msb_bits_grad, msb_bits_weight, threshold, pretrained=False, **kwargs):
    """SkipNet-74 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_bits, num_bits_weight, num_bits_bias,
                                  num_bits_grad, msb_bits, msb_bits_grad, msb_bits_weight, threshold,
                                  num_classes=10, embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_110(pretrained=False,  **kwargs):
    """SkipNet-110 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_152(pretrained=False,  **kwargs):
    """SkipNet-152 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [25, 25, 25], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


# For CIFAR-100
def cifar100_rnn_gate_38(pretrained=False, **kwargs):
    """SkipNet-38 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [6, 6, 6], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_74(pretrained=False, **kwargs):
    """SkipNet-74 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_110(pretrained=False, **kwargs):
    """SkipNet-110 with Recurrent Gate """
    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_152(pretrained=False, **kwargs):
    """SkipNet-152 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [25, 25, 25], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model

