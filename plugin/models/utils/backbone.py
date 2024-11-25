from collections import OrderedDict

import torch
import torchvision
from torch import nn


# class Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = torchvision.models.swin_v2_t(weights=torchvision.models.Swin_V2_T_Weights.DEFAULT)
#         self.out_layers = 4
#         self.out_channels = [96, 192, 384, 768]

#     def forward(self, x):
#         outs = []
#         for idx, layer in enumerate(self.backbone):
#             x = layer(x)
#             if idx in (1, 3, 5, 7):
#                 out = x.permute(0, 3, 1, 2).contiguous()
#                 outs.append(out)
#         return outs

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.ones_(self.beta)
        nn.init.ones_(self.tau)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone = torchvision.models.resnet50(norm_layer=FRN)
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT,
                                               norm_layer=FrozenBatchNorm2d)
        layers = OrderedDict()
        for name, layer in backbone.named_children():
            if name != 'avgpool' and name != 'fc':
                layers[name] = layer
        self.backbone = nn.Sequential(layers)

        # 只训练后几层的参数， 使用pretrain参数的时候可以试试
        for name, parameter in self.backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        self.out_layers = 4
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        outs = []
        for name, layer in self.backbone.named_children():
            x = layer(x)
            if name in ('layer1', 'layer2', 'layer3', 'layer4'):
                outs.append(x)
        return outs
