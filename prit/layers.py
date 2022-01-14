from typing import Callable, Union

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block as timm_Block


class Patch(object):
    num_patches: Union[int, Callable]

    @property
    def P(self) -> int:
        if isinstance(self.num_patches, Callable):
            return self.num_patches()
        elif isinstance(self.num_patches, int):
            return self.num_patches
        raise TypeError(f"num_patches must be int or Callable, bug got {type(self.num_patches)}.")


class PatchDownsample(nn.Module, Patch):
    pool: nn.Module
    reduction: nn.Module
    norm: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, PG, D_in = x.shape
        Gh = Gw = int((PG // self.P)**.5)  # stage1: 8*8, stage2: 4*4, stage3: 2*2, stage4: 1*1

        # get grid-like patches
        x = x.reshape(N, self.P, Gh, Gw, D_in)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(N * self.P, D_in, Gh, Gw)

        # downsample
        x= self.pool(x)

        # reshape
        D_out = x.shape[1]
        x = x.reshape(N, self.P, D_out, Gh // 2, Gw // 2)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(N, self.P * Gh // 2 * Gw // 2, D_out)

        x = self.reduction(x)
        x = self.norm(x)
        return x


class PatchPool(PatchDownsample):
    def __init__(self, dim_in, dim_out, stride, norm_layer=nn.LayerNorm, num_patches=None):
        assert stride == 1 or stride == 2
        super().__init__()
        self.pool = nn.AvgPool2d(stride)
        self.reduction = nn.Linear(dim_in, dim_out, bias=False)
        self.norm = norm_layer(dim_out)
        self.num_patches = num_patches


class PatchConv(PatchDownsample):
    def __init__(self, dim_in, dim_out, stride, norm_layer=nn.LayerNorm, num_patches=None):
        assert stride == 1 or stride == 2
        super().__init__()
        self.pool = nn.Conv2d(dim_in, dim_out, stride, stride=stride)
        self.reduction = nn.Identity()
        self.norm = norm_layer(dim_out)
        self.num_patches = num_patches


class PatchDWConv(PatchDownsample):
    def __init__(self, dim_in, dim_out, stride, norm_layer=nn.LayerNorm, num_patches=None):
        assert stride == 1 or stride == 2
        super().__init__()
        self.pool = nn.Conv2d(dim_in, dim_in, stride, stride=stride, groups=dim_in)
        self.reduction = nn.Linear(dim_in, dim_out, bias=False)
        self.norm = norm_layer(dim_out)
        self.num_patches = num_patches


class Block(timm_Block, Patch):
    def __init__(self, *args, num_patches=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = num_patches


class LocalBlock(Block):

    def forward(self, x):
        N, PG, D = x.shape
        G = PG // self.P

        # windowed attention
        x = x.view(N * self.P, G, D)
        x = super().forward(x)
        x = x.view(N, PG, D)

        return x
