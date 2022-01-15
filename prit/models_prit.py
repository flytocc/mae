# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed

from .layers import Block, LocalBlock, PatchPool, PatchConv, PatchOverlapConv, PatchDWConv


BLOCKS = {'normal': Block, 'local': LocalBlock}
PATCH_DOWNSAMPLE = {'pool': PatchPool, 'conv': PatchConv, 'overlap_conv': PatchOverlapConv, 'dwconv': PatchDWConv}


class PyramidReconstructionImageTransformer(nn.Module):
    def __init__(self,
                 # args for ViT (timm)
                 # w/o `embed_dim`, `detph` and `hybrid_backbone`.
                 # default value of `patch_size` and `num_heads` and  changed.
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 # args for PriT
                 strides=(1, 2, 2, 2), depths=(2, 2, 6, 2),
                 dims=(256, 512, 1024, 2048), num_heads=(4, 8, 16, 32),
                 downsamples=('pool', 'pool', 'pool'),
                 blocks=('normal', 'normal', 'normal', 'normal')):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dims[-1]  # num_features for consistency with other models
        self.out_patch_size = patch_size * reduce(mul, strides)
        self.num_out_patches = int((img_size // self.out_patch_size)**2)
        self.split_shape = (img_size // self.out_patch_size, self.out_patch_size // patch_size) * 2

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0])
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dims[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        stages = []
        for i in range(len(depths)):
            downsample = partial(PATCH_DOWNSAMPLE[downsamples[i - 1]], num_patches=self.num_out_patches)
            block = partial(BLOCKS[blocks[i]], num_patches=self.num_out_patches)
            stages.append(nn.Sequential(
                downsample(dims[i - 1], dims[i], strides[i], norm_layer=norm_layer)
                    if i > 0 and (strides[i] != 1 or dims[i - 1] != dims[i]) else nn.Identity(),
                *[block(dims[i], num_heads[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr.pop(0), norm_layer=norm_layer)
                  for _ in range(depths[i])]
            ))
        self.stages = nn.ModuleList(stages)
        self.norm = norm_layer(dims[-1])

        # Classifier head
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @property
    def blocks(self):
        return self.stages  # for layer decay

    def get_classifier(self):
        return self.head

    def grid_patches(self, x, split_shape):
        B, N, L = x.shape
        x = x.reshape(B, *split_shape, L)               # Bx  (7x8x7x8)  xL
        x = x.permute([0, 1, 3, 2, 4, 5])               # Bx   7x7x8x8   xL
        x = x.reshape(B, x.size(1) * x.size(2), -1, L)  # Bx (7x7)x(8x8) xL
        return x

    def forward_features(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = self.pos_drop(x + self.pos_embed)

        # get grid-like patches
        x = self.grid_patches(x, self.split_shape)
        x = x.flatten(1, 2)  # Bx (7x7x8x8) xL

        # apply Transformer stages
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)

        return x.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def prit_small(**kwargs):
    model = PyramidReconstructionImageTransformer(
        patch_size=4, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 2, 2, 2), depths=(2, 2, 7, 1), dims=(96, 192, 384, 768), num_heads=(2, 4, 8, 16),
        **kwargs)
    return model


def prit_base(**kwargs):
    model = PyramidReconstructionImageTransformer(
        patch_size=4, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 2, 2, 2), depths=(2, 2, 7, 1), dims=(192, 384, 768, 1536), num_heads=(3, 6, 12, 24),
        **kwargs)
    return model


def prit_large(**kwargs):
    model = PyramidReconstructionImageTransformer(
        patch_size=4, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 2, 2, 2), depths=(4, 4, 12, 4), dims=(256, 512, 1024, 2048), num_heads=(4, 8, 16, 32),
        **kwargs)
    return model


# set recommended archs
# small
prit_small_LGGG = partial(prit_small,
    blocks=('local', 'normal', 'normal', 'normal'))                                                # 22.09 M
prit_small_LGGG_conv = partial(prit_small,
    downsamples=('overlap_conv', 'conv', 'conv'), blocks=('local', 'normal', 'normal', 'normal'))  # 23.34 M
prit_small_LGGG_dwconv = partial(prit_small,
    downsamples=('dwconv', 'dwconv', 'dwconv'), blocks=('local', 'normal', 'normal', 'normal'))    # 22.09 M

# base
prit_base_LGGG = partial(prit_base,
    blocks=('local', 'normal', 'normal', 'normal'))
prit_base_LGGG_conv = partial(prit_base,
    downsamples=('overlap_conv', 'conv', 'conv'), blocks=('local', 'normal', 'normal', 'normal'))
prit_base_LGGG_dwconv = partial(prit_base,
    downsamples=('dwconv', 'dwconv', 'dwconv'), blocks=('local', 'normal', 'normal', 'normal'))
