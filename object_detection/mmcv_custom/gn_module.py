# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch

import torch.nn as nn

from mmcv.utils import _BatchNorm, _InstanceNorm
from mmcv.cnn.utils import constant_init, kaiming_init
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.conv import build_conv_layer
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.padding import build_padding_layer
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GlobalLocalConv2Norm(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        try:
            x2 = x2 * weight
        except:
            print(x2.shape, weight.shape)
            raise NotImplementedError()
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


@PLUGIN_LAYERS.register_module()
class GNConvModule(nn.Module):
    def __init__(self, 
        in_channels,
        out_channels,
        kernel_size=7,
        dim_ratio=1,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        inplace=False,
        proj_out=False,
        order=2,
        type='gnconv',
        h=32,
        w=17,
    ):
        super().__init__()
        print(f'[gconv] type={type}')
        self.order = order
        dim = out_channels * dim_ratio
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        print(f'[gconv]: kernel size = {kernel_size}')
        print(f'[gconv]: dim ratio = {dim_ratio}')
        print('[gconv]', order, 'order with dims=', self.dims)
        self.proj_in = nn.Conv2d(in_channels, 2 * dim, 1)

        if type == 'gnconv':
            self.dwconv = get_dwconv(sum(self.dims), kernel_size, True) # nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=3, groups=sum(self.dims))
        elif type == 'gngf':
            self.dwconv = GlobalLocalConv2Norm(sum(self.dims), h=h, w=w) # nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=3, groups=sum(self.dims))
        else:
            raise NotImplementedError()

        if proj_out:
            self.proj_out = nn.Conv2d(dim, out_channels, 1)
        else:
            self.proj_out = nn.Identity()

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        # build normalization layers
        if self.with_norm:
            norm_channels = out_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc)

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)

        return x

if __name__ == '__main__':
    fpn_conv = GNConvModule(
        384,
        256,
        3,
        norm_cfg = dict(type='SyncBN', requires_grad=True),
        inplace=False)

    x = torch.randn(10, 384, 224, 224)
    x = fpn_conv(x)
    print(x.shape)
