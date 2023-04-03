""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .timm_model import TimmModel
from .utils import freeze_batch_norm_2d, to_2tuple

import functools
from operator import mul

from torch.nn.modules.utils import _pair

from .loralib import layers as lora_layers
from .loralib.utils import mark_only_lora_as_trainable

from .MultiheadAttentionP import MultiheadAttentionP


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, lora=-1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            lora: int = -1,
            prompts_lora: int = -1,
            prompt_attention: bool = False,
            num_prompts: int = 10,
            prompt_attention_full: bool = False,
            mask_attention: bool = False,
            prompt_inner_attention: bool = False
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_attention_full = prompt_attention_full
        if prompt_attention_full:
            self.ln_1_prompt = LayerNorm(d_model)
        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.mask_attention = mask_attention
        self.prompt_inner_attention = prompt_inner_attention
        if mask_attention:
            self.mask = []
            self.mask.append([False] + [True for i in range (num_prompts)] + [False for i in range(49)])
            for i in range(num_prompts):
                self.mask.append([False for i in range(49 + 1 + num_prompts)])
            for i in range(49):
                self.mask.append([False] + [True for i in range (num_prompts)] + [False for i in range(49)])
            self.mask = torch.BoolTensor(self.mask)
        if prompt_inner_attention:
            self.mask = []
            self.mask.append([False for i in range(49 + 1 + num_prompts)])
            for i in range(num_prompts):
                self.mask.append([True] + [True for k in range (num_prompts)] + [False for k in range(49)])
            for i in range(49):
                self.mask.append([False] + [True for k in range (num_prompts)] + [False for k in range(49)])
            self.mask = torch.BoolTensor(self.mask)
        if lora <= 0:
            if not prompt_attention:
                self.attn = nn.MultiheadAttention(d_model, n_head)
            else:
                self.attn = MultiheadAttentionP(d_model, n_head, num_prompts=num_prompts)
        else:
            if prompt_attention:
                self.attn = lora_layers.MultiheadAttentionPrompts(d_model, n_head, r=lora, num_prompts=num_prompts, prompts_lora=prompts_lora)
            else:
                self.attn = lora_layers.MultiheadAttention(d_model, n_head, r=lora)
        if prompt_attention_full:
            self.ln_attn_prompt = LayerNorm(d_model) if scale_attn else nn.Identity()
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        if prompt_attention_full:
            self.ln_2_prompt = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if lora <= 0 :
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            if self.prompt_attention_full:
                self.mlp_prompt = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
        else:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", lora_layers.Linear(d_model, mlp_width, r=lora)),
                ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
                ("gelu", act_layer()),
                ("c_proj", lora_layers.Linear(mlp_width, d_model, r=lora))
            ]))
            if self.prompt_attention_full:
                if prompts_lora == -1:
                    self.mlp_prompt = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model))]))
                else:
                    self.mlp_prompt = nn.Sequential(OrderedDict([
                    ("c_fc", lora_layers.Linear(d_model, mlp_width,r=prompts_lora)),
                    ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
                    ("gelu", act_layer()),
                    ("c_proj", lora_layers.Linear(mlp_width, d_model,r = prompts_lora))]))
            

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if self.mask_attention or self.prompt_inner_attention:
            mask = self.mask.to(device=x.get_device())
            return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.prompt_attention_full:
            x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))
            x = x + self.mlp(self.ln_2(x))
        else:
            x_cls_patch = torch.cat([x[0,:,:].unsqueeze(dim=0),x[self.num_prompts+1:,:,:]])
            x_cls_patch = self.ln_1(x_cls_patch)
            x_prompts = self.ln_1_prompt(x[1:self.num_prompts + 1,:,:])
            x_new = torch.cat([x_cls_patch[0,:,:].unsqueeze(dim=0),x_prompts,x_cls_patch[1:,:,:]])
            x_new = self.attention(x_new, attn_mask=attn_mask)
            x_cls_patch = torch.cat([x_new[0,:,:].unsqueeze(dim=0),x_new[self.num_prompts+1:,:,:]])
            x_cls_patch = self.ln_attn(x_cls_patch)
            x_prompts = self.ln_attn_prompt(x_new[1:self.num_prompts + 1,:,:])
            x_new = torch.cat([x_cls_patch[0,:,:].unsqueeze(dim=0),x_prompts,x_cls_patch[1:,:,:]])
            x = x + x_new
            x_cls_patch = torch.cat([x[0,:,:].unsqueeze(dim=0),x[self.num_prompts+1:,:,:]])
            x_cls_patch = self.ln_2(x_cls_patch)
            x_cls_patch = self.mlp(x_cls_patch)
            x_prompts = self.mlp_prompt(self.ln_2_prompt(x[1:self.num_prompts + 1,:,:]))
            x_new = torch.cat([x_cls_patch[0,:,:].unsqueeze(dim=0),x_prompts,x_cls_patch[1:,:,:]])
            x = x + x_new

        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, lora: int = -1, prompts_lora: int = -1, prompt_attention: bool = False, num_prompts: int = 0, prompt_attention_full: bool = False,mask_attention: int = -1,prompt_inner_attention: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        if mask_attention == -1:
            mask_list = [False for i in range(layers)]
        else:
            mask_list = [False for i in range(mask_attention)] + [True for l in range(layers - mask_attention)]
            mask_list.reverse()


        resblocks_list = []
        for t in range(layers):
            resblocks_list.append(ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, lora=lora, prompts_lora = prompts_lora, prompt_attention = prompt_attention,num_prompts=num_prompts, prompt_attention_full=prompt_attention_full,mask_attention=mask_list[t],prompt_inner_attention=prompt_inner_attention))
        self.resblocks = nn.ModuleList(resblocks_list)
        #self.resblocks = nn.ModuleList([
        #    ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, lora=lora, prompt_attention = prompt_attention,num_prompts=num_prompts, prompt_attention_full=prompt_attention_full)
        #    for _ in range(layers)
        #])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU,
            lora: int = -1,
            image_lora: bool = False,
            prompts_lora: int = -1,
            object_tokens: int = 0,
            relation_tokens: int = 0,
            prompt_attention: bool = False,
            prompt_attention_full: bool = False,
            mask_attention: int = -1,
            prompt_inner_attention: bool = False
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        if image_lora:
            self.conv1 = lora_layers.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False, r=lora)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.num_tokens = object_tokens + relation_tokens

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)
        if image_lora:
            self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer, lora=lora, prompts_lora=prompts_lora, prompt_attention = prompt_attention,num_prompts = self.num_tokens, prompt_attention_full=prompt_attention_full,mask_attention=mask_attention,prompt_inner_attention=prompt_inner_attention)
        else:
            self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer, lora=-1,prompts_lora=-1, prompt_attention = prompt_attention,num_prompts = self.num_tokens,prompt_attention_full=prompt_attention_full,mask_attention=mask_attention,prompt_inner_attention=prompt_inner_attention)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.object_tokens = object_tokens
        if self.object_tokens > 0:
            val = math.sqrt(6. / float(3 * functools.reduce(mul, _pair(patch_size), 1) + width))  #prompt init per visual prompt tuning

            self.object_prompts = nn.Parameter(torch.zeros(1, object_tokens, width))
            # xavier_uniform initialization
            nn.init.uniform_(self.object_prompts, -val, val) 

        self.relation_tokens = relation_tokens
        if self.relation_tokens > 0:
            val = math.sqrt(6. / float(3 * functools.reduce(mul, _pair(patch_size), 1) + width))  #prompt init per visual prompt tuning

            self.relation_prompts = nn.Parameter(torch.zeros(1, relation_tokens, width))
            # xavier_uniform initialization
            nn.init.uniform_(self.relation_prompts, -val, val) 

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        if self.object_tokens > 0 and self.relation_tokens > 0:
            x = torch.cat([x[:, :1, :],self.object_prompts.expand(x.shape[0], -1, -1), self.relation_prompts.expand(x.shape[0], -1, -1),x[:, 1:, :]], dim=1)
        elif self.object_tokens > 0:
            x = torch.cat([x[:, :1, :],self.object_prompts.expand(x.shape[0], -1, -1),x[:, 1:, :]], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x1 = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x1 = x1 @ self.proj

        x2 = x[:,1:1 + self.num_tokens,:]

        return x1, x2


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12


class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            lora: int = -1,
            image_lora: bool = False,
            text_lora: bool = False,
            prompts_lora: int = -1,
            object_tokens: int = 0,
            relation_tokens: int = 0,
            prompt_attention: bool = False,
            prompt_attention_full: bool = False,
            mask_attention: int = -1,
            prompt_inner_attention: bool = False
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size
            )
            act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=embed_dim,
                act_layer=act_layer,
                lora=lora,
                image_lora = image_lora,
                prompts_lora = prompts_lora,
                object_tokens=object_tokens,
                relation_tokens=relation_tokens,
                prompt_attention = prompt_attention,
                prompt_attention_full = prompt_attention_full,
                mask_attention=mask_attention,
                prompt_inner_attention=prompt_inner_attention
            )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
            lora=lora if text_lora else -1
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)


        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features, object_tokens = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, object_tokens, text_features, self.logit_scale.exp()


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_from_openai_state_dict(state_dict: dict, lora: int = -1, image_lora: bool = False, text_lora: bool = False, prompts_lora: int = -1, object_tokens: int = 0, relation_tokens: int = 0, prompt_attention: bool = False, prompt_attention_full: bool = False, mask_attention: int = -1,prompt_inner_attention: bool = False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
        lora=lora,
        image_lora = image_lora,
        text_lora = text_lora,
        prompts_lora = prompts_lora,
        object_tokens=object_tokens,
        relation_tokens=relation_tokens,
        prompt_attention = prompt_attention,
        prompt_attention_full = prompt_attention_full,
        mask_attention=mask_attention,
        prompt_inner_attention=prompt_inner_attention
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict, strict=False)
    if prompts_lora != -1:
        with torch.no_grad():
            for b in model.visual.transformer.resblocks:
                b.ln_1_prompt.weight.copy_(b.ln_1.weight)
                b.ln_1_prompt.bias.copy_(b.ln_1.bias)
                b.ln_2_prompt.weight.copy_(b.ln_2.weight)
                b.ln_2_prompt.bias.copy_(b.ln_2.bias)
                b.mlp_prompt.c_fc.weight.copy_(b.mlp.c_fc.weight)
                b.mlp_prompt.c_fc.bias.copy_(b.mlp.c_fc.bias)
                b.mlp_prompt.c_proj.weight.copy_(b.mlp.c_proj.weight)
                b.mlp_prompt.c_proj.bias.copy_(b.mlp.c_proj.bias)
                b.attn.in_prompts_proj_weight.copy_(b.attn.in_proj_weight)
                b.attn.in_prompts_proj_bias.copy_(b.attn.in_proj_bias)
                b.attn.out_prompts_proj.weight.copy_(b.attn.out_proj.weight)
                b.attn.out_prompts_proj.bias.copy_(b.attn.out_proj.bias)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
