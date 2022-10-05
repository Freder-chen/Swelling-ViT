# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# --------------------------------------------------------

import math
import logging
import numpy as np
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.vision_transformer import (
    build_model_with_cfg, named_apply, adapt_input_conv,
    PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
)

from util.pos_embed import get_2d_sincos_pos_embed


_logger = logging.getLogger(__name__)


__all__ = [
    'swelling_vit_tiny_patch2_32', 'swelling_vit_small_patch2_32', 'swelling_vit_base_patch2_32',
    'swelling_vit_tiny_patch16_224', 'swelling_vit_small_patch16_224', 'swelling_vit_base_patch16_224',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'swelling_vit_tiny_patch2_32': _cfg(url='', input_size=(3, 32, 32)),
    'swelling_vit_small_patch2_32': _cfg(url='', input_size=(3, 32, 32)),
    'swelling_vit_base_patch2_32': _cfg(url='', input_size=(3, 32, 32)),
    'swelling_vit_tiny_patch16_224': _cfg(url=''),
    'swelling_vit_small_patch16_224': _cfg(url=''),
    'swelling_vit_base_patch16_224': _cfg(url=''),
}


def generate_list(start, num=12, step=0):
    return [start + i * step for i in range(num)]


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_ntuple = _ntuple


class SwellAttention(nn.Module):
    def __init__(
        self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.,
        att_distance=0, num_patches=196, num_tokens=0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        att_distance = to_ntuple(num_heads)(att_distance)
        
        self.attention_distance = att_distance
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.table_size = (2 * int(math.sqrt(num_patches)) - 1) ** 2 + num_tokens  # (2 * h - 1) * (2 * w - 1) + 0/1

        relative_position_index = self.generate_relative_position_index(num_patches, num_tokens=num_tokens)
        self.register_buffer('relative_position_index', relative_position_index)

        relative_position_table = torch.zeros(self.table_size, num_heads)
        trunc_normal_(relative_position_table, std=.02)
        self.register_parameter('relative_position_table', nn.Parameter(relative_position_table))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask_ratio=0.):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # relative attention
        relative_bias = self.relative_position_table[self.relative_position_index.view(-1)].view(N, N, -1)  # h*w, h*w, nH
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
        attn = attn + relative_bias

        # swell receptive field
        if self.training:
            relative_masking_table = torch.zeros(B, self.num_heads, self.table_size, device=x.device)  # H, T
            # mask relative table
            for i in range(self.num_heads):
                init_distance = self.attention_distance[i]
                max_distance = int(math.sqrt(self.num_patches)) - 1
                current_distance = int((max_distance - init_distance + 1) * (1 - mask_ratio) + init_distance)
                masking_index = self.masking_non_local(B, current_distance, self.num_patches, self.num_tokens)  # b, n
                relative_masking_table[:, i, :].scatter_(dim=1, index=masking_index.to(x.device), value=-100, reduce='add')
            # rel to abs
            masking_table = torch.gather(relative_masking_table, 2, self.relative_position_index.view(-1).unsqueeze(0).unsqueeze(0).repeat(B, self.num_heads, 1))  # B, nH, h*w*h*w
            masking_table = masking_table.view(B, -1, N, N).contiguous()  # B, nH, h*w, h*w
            attn = attn + masking_table  # B, nH, h*w, h*w

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        return x
    
    @staticmethod
    def masking_non_local(batch_size, attention_distance, num_patches, num_tokens=0):
        """
        return: [b, n]
        """
        size = 2 * int(math.sqrt(num_patches)) - 1
        coords_h, coords_w = torch.arange(size), torch.arange(size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w

        # masking by distance
        coords_distance = torch.maximum(torch.abs(coords[0, :] - size // 2), torch.abs(coords[1, :] - size // 2))  # h, w
        masking_coords = (coords_distance > attention_distance).nonzero(as_tuple=False)  # n, 2
        masking_coords = masking_coords.repeat(batch_size, 1, 1)

        return masking_coords[:, :, 0] * size + masking_coords[:, :, 1] + num_tokens  # b, n
    
    @staticmethod
    def generate_relative_position_index(num_patches, num_tokens=0):
        size = int(math.sqrt(num_patches))
        coords_h, coords_w = torch.arange(size), torch.arange(size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, h*w
        relative_coords = coords_flatten[:, None, :] - coords_flatten[:, :, None]  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += size - 1  # shift to start from 0
        relative_coords[:, :, 1] += size - 1
        relative_coords[:, :, 0] *= 2 * size - 1
        relative_position_index = relative_coords.sum(-1)  # h*w, h*w
        if num_tokens > 0:
            relative_position_index += num_tokens
            temp = torch.zeros(num_patches + num_tokens, num_patches + num_tokens)
            temp[num_tokens:, num_tokens:] = relative_position_index
            relative_position_index = temp
        return relative_position_index.long()


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 att_distance=0, num_patches=196, num_tokens=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SwellAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
            att_distance=att_distance, num_patches=num_patches, num_tokens=num_tokens
        )
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_ratio=0):
        x = x + self.drop_path(self.attn(self.norm1(x), mask_ratio))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., representation_size=None, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', global_pool=False, attention_distance=None):
        super().__init__()
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                att_distance=attention_distance[i], num_patches=num_patches, num_tokens=self.num_tokens
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # trunc_normal_(self.pos_embed, std=.02)

        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 'cls_token',
            *{f'blocks.{i}.attn.relative_position_table' for i in range(len(self.blocks))},
        }

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, mask_ratio)

        x = self.norm(x)
        x = x.mean(dim=1) if self.global_pool else x[:, 0]
        
        return self.pre_logits(x)

    def forward(self, x, mask_ratio=0):
        x = self.forward_features(x, mask_ratio)
        x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        pretrained_strict=False,
        **kwargs)
    return model


def swelling_vit_tiny_patch2_32(pretrained=False, **kwargs):
    attention_distance = [generate_list(1, 3, 0)] * 12
    model_kwargs = dict(
        patch_size=2, embed_dim=192, depth=12, num_heads=3, 
        attention_distance=attention_distance, **kwargs)
    model = _create_vision_transformer('swelling_vit_tiny_patch2_32', pretrained=pretrained, **model_kwargs)
    return model


def swelling_vit_small_patch2_32(pretrained=False, **kwargs):
    attention_distance = [generate_list(1, 6, 0)] * 12
    model_kwargs = dict(
        patch_size=2, embed_dim=384, depth=12, num_heads=6, 
        attention_distance=attention_distance, **kwargs)
    model = _create_vision_transformer('swelling_vit_small_patch2_32', pretrained=pretrained, **model_kwargs)
    return model


def swelling_vit_base_patch2_32(pretrained=False, **kwargs):
    attention_distance = [generate_list(1, 12, 0)] * 12
    model_kwargs = dict(
        patch_size=2, embed_dim=768, depth=12, num_heads=12, 
        attention_distance=attention_distance, **kwargs)
    model = _create_vision_transformer('swelling_vit_base_patch2_32', pretrained=pretrained, **model_kwargs)
    return model


def swelling_vit_tiny_patch16_224(pretrained=False, **kwargs):
    attention_distance = [generate_list(1, 3, 0)] * 12
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        attention_distance=attention_distance, **kwargs)
    model = _create_vision_transformer('swelling_vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def swelling_vit_small_patch16_224(pretrained=False, **kwargs):
    attention_distance = [generate_list(1, 6, 0)] * 12
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        attention_distance=attention_distance, **kwargs)
    model = _create_vision_transformer('swelling_vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def swelling_vit_base_patch16_224(pretrained=False, **kwargs):
    attention_distance = [generate_list(1, 12, 0)] * 12
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        attention_distance=attention_distance, **kwargs)
    model = _create_vision_transformer('swelling_vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# def swelling_vit_large_patch16_224(pretrained=False, **kwargs):
#     attention_distance = [generate_list(1, 16, 0)] * 24
#     model_kwargs = dict(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         attention_distance=attention_distance, **kwargs)
#     model = _create_vision_transformer('swelling_vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model
