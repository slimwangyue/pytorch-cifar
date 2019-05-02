'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable


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
                    if x_in is not None and w is not None:
                        x_grad = torch.nn.grad.conv2d_input(x_in.shape, w, grads[0], stride=conv2d_obj.stride, padding=conv2d_obj.padding)
                    if x_in is not None and w is not None:
                        w_grad = torch.nn.grad.conv2d_weight(x_in, w.shape, grads[0], stride=conv2d_obj.stride, padding=conv2d_obj.padding)
                    if b is not None:
                        bias_grad = torch.ones(b.shape, device=torch.device('cuda:0')) * torch.sum(grads[0], dim=(0, 2, 3))
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
