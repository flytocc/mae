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

from timm.models.vision_transformer import PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed

from .layers import Block, LocalBlock, PatchPool, PatchConv, PatchDWConv


BLOCKS = {'normal': Block, 'local': LocalBlock}
PATCH_DOWNSAMPLE = {'pool': PatchPool, 'conv': PatchConv, 'dwconv': PatchDWConv}


class MaskedAutoencoderPriT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 # args for PriT
                 strides=(1, 2, 2, 2), depths=(2, 2, 6, 2), dims=(256, 512, 1024, 2048),
                 num_heads=(4, 8, 16, 32), patch_downsample='pool',
                 blocks=('normal', 'normal', 'normal', 'normal')):
        super().__init__()
        self.out_patch_size = patch_size * reduce(mul, strides)
        self.num_out_patches = int((img_size // self.out_patch_size)**2)
        self.split_shape = (img_size // self.out_patch_size, self.out_patch_size // patch_size) * 2

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dims[0])
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dims[0]), requires_grad=False)  # fixed sin-cos embedding

        stages = []
        downsample = partial(PATCH_DOWNSAMPLE[patch_downsample], num_patches=lambda:self.num_visible)
        for i in range(len(depths)):
            block = partial(BLOCKS[blocks[i]], num_patches=lambda:self.num_visible)
            stages.append(nn.Sequential(
                downsample(dims[i - 1], dims[i], strides[i], norm_layer=norm_layer)
                    if i > 0 and (strides[i] != 1 or dims[i - 1] != dims[i]) else nn.Identity(),
                *[block(dims[i], num_heads[i], mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                  for _ in range(depths[i])]
            ))
        self.stages = nn.ModuleList(stages)
        self.norm = norm_layer(dims[-1])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(dims[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_out_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.out_patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_out_patches**.5))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def num_visible(self):
        return int(self.num_out_patches * (1 - self.mask_ratio))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, out_patch_size**2 *3)
        """
        p = self.out_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, out_patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.out_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def grid_patches(self, x, split_shape):
        N, _, D = x.shape
        x = x.reshape(N, *split_shape, D)               # N, Lh, Gh, Lw, Gw, D
        x = x.permute(0, 1, 3, 2, 4, 5)                 # N, Lh, Lw, Gh, Gw, D
        x = x.reshape(N, x.size(1) * x.size(2), -1, D)  # N, L, G, D
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # set mask_ratio
        self.mask_ratio = mask_ratio

        # get grid-like patches
        x = self.grid_patches(x, self.split_shape)

        N, L, G, D = x.shape  # batch, length, grid, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, G, D))
        x_masked = x_masked.reshape(N, -1, D)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer stages
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_small_patch16_dec192d4b(**kwargs):
    model = MaskedAutoencoderPriT(
        patch_size=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 1, 1, 1), depths=(2, 2, 6, 2), dims=(384,384,384,384), num_heads=(6, 6, 6, 6),
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=4, **kwargs)
    return model


def mae_prit_small_patch16_dec192d4b(**kwargs):
    model = MaskedAutoencoderPriT(
        patch_size=4, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 2, 2, 2), depths=(2, 2, 6, 2), dims=(96, 192, 384, 768), num_heads=(2, 4, 8, 16),
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=4, **kwargs)
    return model


def mae_prit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderPriT(
        patch_size=4, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        strides=(1, 2, 2, 2), depths=(2, 2, 6, 2), dims=(192, 384, 768, 1536), num_heads=(3, 6, 12, 24),
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, **kwargs)
    return model


def mae_prit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderPriT(
        patch_size=4, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        strides=(1, 2, 2, 2), depths=(4, 4, 12, 4), dims=(256, 512, 1024, 2048), num_heads=(4, 8, 16, 32),
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec192d4b  # decoder: 192 dim, 4 blocks
mae_prit_small_patch16 = mae_prit_small_patch16_dec192d4b  # decoder: 192 dim, 4 blocks
mae_prit_base_patch16 = mae_prit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_prit_large_patch16 = mae_prit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
