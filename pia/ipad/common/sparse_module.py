# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


from __future__ import print_function

import time
from operator import itemgetter
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import copy
import types
import random
import math
import warnings
import os
import sys
import pickle
import json
import shutil
import pandas as pd

import numpy as np
from rouge_score import rouge_scorer

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.autograd import Variable
import torch.nn.functional as F

from transformers.activations import gelu
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import rotate_half, apply_rotary_pos_emb

torch.manual_seed(7)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SparseModule(torch.nn.Module):

    def get(self, attr):
        if attr is None:
            return None
        return getattr(self, attr)

    def cp(self, layer, name):
        if name is None:
            return
        setattr(self, name, getattr(layer, name))

    def set(self, name, obj):
        if name is None:
            return
        setattr(self, name, obj)

    def ill(self, name, weight, bias=None):   # init_linear_layer
        out_feature, in_feature = weight.shape
        layer = nn.Linear(in_features=in_feature,out_features=out_feature,bias=bias is None)
        layer.weight.data = weight
        if bias is not None:
            layer.bias.data = bias
        setattr(self, name, layer)

    def calc_input_sensitive(self):
        pass

    def eval_input(self, mask):
        pass

    def reparam_input(self,mask):
        pass


class SparseNorm(SparseModule):
    weight_name = 'weight'
    bias_name = None
    eps_name = 'eps'

    def __init__(self, layer=None, layer_index=0):
        super().__init__()

    def eval_input(self, mask):
        weight = self.get(self.weight_name)
        w = Parameter(weight[mask].contiguous().detach(),
                                                requires_grad=False)
        self.set(self.weight_name, w)
        if self.bias_name is not None:
            bias = self.get(self.bias_name)
            b = Parameter(bias[mask].contiguous().detach(),
                          requires_grad=False)
            self.set(self.bias_name, b)
        m = Parameter(mask[mask], requires_grad=False)
        self.mask = m


class SparseRMSNorm(SparseNorm):
    weight_name = 'weight'
    bias_name = None
    eps_name = 'variance_epsilon'

    def __init__(self, layer=None, mask=None):
        super().__init__()
        self.cp(layer, self.weight_name)
        self.cp(layer, self.bias_name)
        self.cp(layer, self.eps_name)

        self.normalized_shape = self.get(self.weight_name).shape

        self.mask = mask
        self.training = True

    def forward(self, hidden_states):
        weight = self.get(self.weight_name)
        eps = self.get(self.eps_name)
        if self.training:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states[:, :, self.mask == 1].pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            return weight * hidden_states.to(input_dtype)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            return weight * hidden_states.to(input_dtype)

    def eval(self):
        mask = self.mask == 1.0
        weight = self.get(self.weight_name)
        w = Parameter(weight[mask].contiguous().detach(), requires_grad=False)
        self.set(self.weight_name, w)

        self.mask = Parameter(self.mask[mask], requires_grad=False)
        self.normalized_shape = w.shape
        self.training = False


class SparseLayerNorm(SparseNorm):
    weight_name = 'weight'
    bias_name = 'bias'
    eps_name = 'eps'

    def __init__(self, layer=None, mask=None):
        super().__init__()
        self.cp(layer, self.weight_name)
        self.cp(layer, self.bias_name)
        self.cp(layer, self.eps_name)

        self.normalized_shape = layer.normalized_shape
        self.mask = mask
        self.training = True

    def forward(self, hidden_states):
        weight = self.get(self.weight_name)
        bias = self.get(self.bias_name)
        eps = self.get(self.eps_name)

        if self.training:
            std, mean = torch.std_mean(hidden_states[..., self.mask == 1], -1, keepdim=True)
            return (hidden_states - mean) / (std + eps) * weight + bias
        else:
            return torch.layer_norm(hidden_states, self.normalized_shape, weight=weight, bias=bias,
                                    eps=eps)

    def eval(self):
        mask = self.mask == 1.0
        weight = self.get(self.weight_name)
        bias = self.get(self.bias_name)

        w = Parameter(
            weight[mask].contiguous().detach(), requires_grad=False)
        self.set(self.weight_name, w)

        b = Parameter(bias[mask].contiguous().detach(),
                                                       requires_grad=False)
        self.set(self.bias_name, b)

        self.mask = Parameter(self.mask[mask], requires_grad=False)
        self.normalized_shape = w.shape
        self.training = False


class SparseMLP(SparseModule):
    up_name = 'up_proj'
    down_name = 'down_proj'
    gate_name = 'gate_proj'

    def __init__(self, layer=None, layer_index=0):
        super().__init__()

    def set_act(self):
        pass



class GptSparseMLP(SparseMLP):
    up_name = 'dense_h_to_4h'
    down_name = 'dense_4h_to_h'
    gate_name = None

    def __init__(self, layer=None, layer_index=0):
        super().__init__()
        # self.layer = layer  # remove from state dict
        self.layer_index = layer_index

        self.cp(layer, self.up_name)
        self.cp(layer, self.down_name)
        self.set_act()

        layer = self.get(self.up_name)
        self.device = layer.weight.device
        self.dtype = layer.weight.dtype
        self.mask = Parameter(torch.ones(layer.weight.size(0)).to(dtype=self.dtype, device=self.device),
                              requires_grad=False)
        self.alpha = 1.0
        self.training = True

    def forward(self, x):
        up = self.get(self.up_name)
        down = self.get(self.down_name)
        hidden = up(x)
        if self.training:
            act = self.act(hidden) * self.mask
        else:
            act = self.act(hidden)
        vec = down(act)
        return vec

    def update_mask(self, zero_count=1, max_zero_count=1024):
        down = self.get(self.down_name)
        amp = torch.abs(torch.sum(down.weight.grad * down.weight, dim=0))
        if amp.size(0) < 1024:
            print('SparseMLP.amp.shape', amp.shape)
        topk = torch.topk(amp, max(zero_count, 1), largest=False)
        if torch.sum(1 - self.mask) < max_zero_count:
            self.mask[topk.indices] = 0.0
        mask_counts = [torch.sum(1 - self.mask).item()]
        topk_values = [torch.mean(topk.values).item()]
        max_grads = [torch.max(down.weight.grad).item()]
        return mask_counts, topk_values, max_grads

    def calc_input_sensitive(self):
        l1 = self.get(self.up_name)
        amp = torch.sum(l1.weight.grad * l1.weight, dim=0)
        max_grads = [torch.max(l1.weight.grad).item()]
        return amp, max_grads

    def eval(self):
        mask = self.mask.data
        count = torch.sum(mask).item()
        pad_count = count - count // 16 * 16
        mask_value = mask.float().cpu().numpy()
        pad_indices = []
        if pad_count > 0:
            for j, v in mask_value:
                if v == 0:
                    pad_indices.append(j)
                if len(pad_indices) >= pad_count:
                    break
            pad_indices = torch.from_numpy(np.array(pad_indices, dtype=np.int32)).to(self.device)
            mask[pad_indices] = 1.0
            # self.log(f'padding {pad_count} from {count} to {count + pad_count}')
            count = count + pad_count
        # self.log(f'shrink from {mask.size(0)} to {count}')
        l1 = self.get(self.up_name)
        l1.weight.requires_grad = False
        l1.bias.requires_grad = False
        l1.weight[pad_indices] = 0.0
        l1.bias[pad_indices] = 0.0
        l1.weight = Parameter(l1.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        l1.bias = Parameter(l1.bias[mask == 1.0].contiguous().detach(), requires_grad=False)
        l1.out_features = l1.bias.size(0)
        l2 = self.get(self.down_name)
        l2.weight.requires_grad = False
        l2.bias.requires_grad = False
        weight = l2.weight.t()
        weight[pad_indices] = 0.0
        l2.bias[pad_indices] = 0.0
        l2.weight = Parameter(weight[mask == 1.0].t().contiguous().detach(), requires_grad=False)
        l2.bias = Parameter(l2.bias.detach(), requires_grad=False)
        l2.in_features = l1.out_features
        self.mask = Parameter(mask[mask == 1.0], requires_grad=False)
        self.training = False

    def reparam(self, coef=0.001):
        mask = self.mask.data.float().cpu().numpy()
        reparam_indices = np.nonzero(mask)[0]
        reparam_count = len(reparam_indices)
        up = self.get(self.up_name)
        down = self.get(self.down_name)
        d = up.weight.size(1)
        up.weight.data[reparam_indices] = coef * torch.randn(reparam_count, d).to(dtype=self.dtype,
                                                                                                  device=self.device)
        if up.bias is not None:
            up.bias.data[reparam_indices] = coef * torch.randn(reparam_count).to(dtype=self.dtype,
                                                                                             device=self.device)
        down.weight.data[:, reparam_indices] = coef * torch.randn(d, reparam_count).to(dtype=self.dtype,
                                                                                                     device=self.device)
        self.mask[reparam_indices] = 1.0

    def set_act(self):
        self.act = gelu

    def reparam_input(self, mask, coef=0.001):
        mask = mask.float().cpu().numpy()
        reparam_indices = np.nonzero(mask)[0]
        reparam_count = len(reparam_indices)
        if reparam_count == 0:
            return

        # MLP
        l1 = self.get(self.up_name)
        out_feauture, in_feature = l1.weight.shape
        l1.weight.data[:, reparam_indices] = coef * torch.randn(out_feauture, reparam_count).to(dtype=self.dtype,
                                                                                                device=self.device)

        l2 = self.get(self.down_name)
        out_feauture, in_feature = l2.weight.shape
        l2.weight.data[reparam_indices] = coef * torch.randn(reparam_count, in_feature).to(dtype=self.dtype,
                                                                                           device=self.device)
        l2.bias.data[reparam_indices] = coef * torch.randn(reparam_count).to(dtype=self.dtype, device=self.device)


class LlamaSparseMLP(SparseMLP):
    up_name = 'up_proj'
    down_name = 'down_proj'
    gate_name = 'gate_proj'

    def __init__(self, layer=None, layer_index=0):
        super().__init__()
        self.layer_index = layer_index

        self.cp(layer, self.up_name)
        self.cp(layer, self.down_name)
        self.cp(layer, self.gate_name)
        self.set_act()

        layer = self.get(self.up_name)
        self.device = layer.weight.device
        self.dtype = layer.weight.dtype
        self.dim = layer.weight.size(0)
        self.mask = Parameter(torch.ones(self.dim).to(dtype=self.dtype, device=self.device), requires_grad=False)
        self.alpha = 1.0
        self.training = True

    def forward(self, x):
        up = self.get(self.up_name)
        gate = self.get(self.gate_name)
        down = self.get(self.down_name)
        if self.training:
            vec = down(self.act(gate(x)) * up(x) * self.mask)
        else:
            vec = down(self.act(gate(x)) * up(x))
        return vec

    def update_mask(self, zero_count=1, max_zero_count=1024):
        down = self.get(self.down_name)
        amp = torch.abs(torch.sum(down.weight.grad * down.weight, dim=0))
        topk = torch.topk(amp, max(zero_count, 1), largest=False)
        if torch.sum(1 - self.mask) < max_zero_count:
            self.mask[topk.indices] = 0.0
        mask_counts = [torch.sum(1 - self.mask).item()]
        topk_values = [torch.mean(topk.values).item()]
        max_grads = [torch.max(down.weight.grad).item()]
        return mask_counts, topk_values, max_grads

    def eval(self):
        mask = self.mask.data
        count = torch.sum(mask).item()
        pad_count = count - count // 16 * 16
        mask_value = mask.float().cpu().numpy()
        pad_indices = []
        if pad_count > 0:
            for j, v in mask_value:
                if v == 0:
                    pad_indices.append(j)
                if len(pad_indices) >= pad_count:
                    break
            pad_indices = torch.from_numpy(np.array(pad_indices, dtype=np.int32)).to(self.device)
            mask[pad_indices] = 1.0
            # self.log(f'padding {pad_count} from {count} to {count + pad_count}')
            count = count + pad_count
        # self.log(f'shrink from {mask.size(0)} to {count}')
        l1 = self.get(self.up_name)
        l1.weight.requires_grad = False
        l1.weight[pad_indices] = 0.0
        l1.weight = Parameter(l1.weight[mask == 1.0].contiguous().detach(), requires_grad=False)

        l1 = self.get(self.gate_name)
        l1.weight.requires_grad = False
        l1.weight[pad_indices] = 0.0
        l1.weight = Parameter(l1.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        l1.out_features = l1.weight.size(0)

        l2 = self.get(self.down_name)
        l2.weight.requires_grad = False
        weight = l2.weight
        weight[:, pad_indices] = 0.0
        l2.weight = Parameter(weight[:, mask == 1.0].contiguous().detach(), requires_grad=False)
        l2.in_features = l1.out_features
        self.mask = Parameter(mask[mask == 1.0], requires_grad=False)
        self.training = False

    def reparam(self, coef=0.001):
        mask = self.mask.data.float().cpu().numpy()
        reparam_indices = np.nonzero(mask)[0]
    
        reparam_count = len(reparam_indices)
        if reparam_count == 0:
            return
        
        up = self.get(self.up_name)
        gate = self.get(self.gate_name)
        down = self.get(self.down_name)
        d = up.weight.size(1)
        up.weight.data[reparam_indices] = coef * torch.randn(reparam_count, d).to(dtype=self.dtype,
                                                                                            device=self.device)
        gate.weight.data[reparam_indices] = coef * torch.randn(reparam_count, d).to(dtype=self.dtype,
                                                                                              device=self.device)
        down.weight.data[:, reparam_indices] = coef * torch.randn(d, reparam_count).to(dtype=self.dtype,
                                                                                                 device=self.device)
        self.mask[reparam_indices] = 1.0

    def set_act(self):
        self.act = F.silu

    def calc_input_sensitive(self):
        max_grads = []
        amp = 0.0
        l1 = self.get(self.up_name)
        amp1 = torch.sum(l1.weight.grad * l1.weight, dim=0)
        max_grads.append(torch.max(l1.weight.grad).item())
        amp += amp1

        l1 = self.get(self.gate_name)
        amp1 = torch.sum(l1.weight.grad * l1.weight, dim=0)
        max_grads.append(torch.max(l1.weight.grad).item())
        amp += amp1

        return amp, max_grads

    def eval_input(self, mask):
        l1 = self.get(self.up_name)
        l1.weight.requires_grad = False
        l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        l1.in_features = l1.weight.size(1)

        l1 = self.get(self.gate_name)
        l1.weight.requires_grad = False
        l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        l1.in_features = l1.weight.size(1)

        l2 = self.get(self.down_name)
        l2.weight.requires_grad = False
        l2.weight = Parameter(l2.weight[mask].contiguous().detach(), requires_grad=False)
        if l2.bias is not None:
            l2.bias.requires_grad = False
            l2.bias = Parameter(l2.bias[mask].detach(), requires_grad=False)
        l2.out_features = l1.in_features

    def reparam_input(self,mask, coef=0.001):

        mask = mask.float().cpu().numpy()
        reparam_indices = np.nonzero(mask)[0]
        reparam_count = len(reparam_indices)
        if reparam_count == 0:
            return
        
        l1 = self.get(self.up_name)
        out_feauture, in_feature = l1.weight.shape
        l1.weight.data[:, reparam_indices] = coef * torch.randn(out_feauture, reparam_count).to(dtype=self.dtype,
                                                                                                device=self.device)

        l1 = self.get(self.gate_name)
        out_feauture, in_feature = l1.weight.shape
        l1.weight.data[:, reparam_indices] = coef * torch.randn(out_feauture, reparam_count).to(dtype=self.dtype,
                                                                                                device=self.device)

        l2 = self.get(self.down_name)
        out_feauture, in_feature = l2.weight.shape
        l2.weight.data[reparam_indices] = coef * torch.randn(reparam_count, in_feature).to(dtype=self.dtype,
                                                                                           device=self.device)
        if l2.bias is not None:
            l2.bias.data[reparam_indices] = coef * torch.randn(reparam_count).to(dtype=self.dtype, 
                                                                                device=self.device)


class SparseAttn(SparseModule):
    q_name = 'q_proj'
    k_name = 'k_proj'
    v_name = 'v_proj'
    qkv_name = None
    o_name = 'o_proj'

    def __init__(self, layer=None, layer_index=0):
        super().__init__()
        self.layer_index = layer_index

    def set_dim(self, layer):
        pass

    def calc_input_sensitive(self):
        amp = 0.0
        max_grads = []
        l1 = self.get(self.q_name)
        amp2 = torch.sum(l1.weight.grad * l1.weight, dim=0)
        max_grads.append(torch.max(l1.weight.grad).item())
        amp += amp2

        l1 = self.get(self.k_name)
        amp2 = torch.sum(l1.weight.grad * l1.weight, dim=0)
        max_grads.append(torch.max(l1.weight.grad).item())
        amp += amp2

        l1 = self.get(self.v_name)
        amp2 = torch.sum(l1.weight.grad * l1.weight, dim=0)
        max_grads.append(torch.max(l1.weight.grad).item())
        amp += amp2
        return amp, max_grads

    def eval_input(self, mask):
        l1 = self.get(self.q_name)
        l1.weight.requires_grad = False
        l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        l1.in_features = l1.weight.size(1)

        l1 = self.get(self.k_name)
        l1.weight.requires_grad = False
        l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        l1.in_features = l1.weight.size(1)

        l1 = self.get(self.v_name)
        l1.weight.requires_grad = False
        l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        l1.in_features = l1.weight.size(1)

        l2 = self.get(self.o_name)
        l2.weight.requires_grad = False
        l2.weight = Parameter(l2.weight[mask].contiguous().detach(), requires_grad=False)
        if l2.bias is not None:
            l2.bias.requires_grad = False
            l2.bias = Parameter(l2.bias[mask].detach(), requires_grad=False)
        l2.out_features = l1.in_features

    def reparam_input(self, mask):
        mask = mask.float().cpu().numpy()
        reparam_indices = np.nonzero(mask)[0]
        reparam_count = len(reparam_indices)
        if reparam_count == 0:
            return

        # ATTN
        l1 = self.get(self.q_name)
        out_feauture, in_feature = l1.weight.shape
        l1.weight.data[:, reparam_indices] = 0.001 * torch.rand(out_feauture, reparam_count).to(dtype=self.dtype,
                                                                                                device=self.device)

        l1 = self.get(self.k_name)
        out_feauture, in_feature = l1.weight.shape
        l1.weight.data[:, reparam_indices] = 0.001 * torch.rand(out_feauture, reparam_count).to(dtype=self.dtype,
                                                                                                device=self.device)

        l1 = self.get(self.v_name)
        out_feauture, in_feature = l1.weight.shape
        l1.weight.data[:, reparam_indices] = 0.001 * torch.rand(out_feauture, reparam_count).to(dtype=self.dtype,
                                                                                                device=self.device)

        l2 = self.get(self.o_name)
        out_feauture, in_feature = l2.weight.shape
        l2.weight.data[reparam_indices] = 0.001 * torch.rand(reparam_count, in_feature).to(dtype=self.dtype,
                                                                                           device=self.device)
        if l2.bias is not None:
            l2.bias.data[reparam_indices] = 0.001 * torch.rand(reparam_count).to(dtype=self.dtype, device=self.device)

        mask[reparam_indices] = 1.0


    def reparam(self):

        n_head = self.num_heads // self.n_repeat
        d_head = self.head_dim

        mask = self.mask[0].float().cpu().numpy()
        q_reparam_indices = np.nonzero(1-mask)[0]
        query_mask_count = len(q_reparam_indices)
        masks = np.reshape(mask, [self.num_heads, d_head])
        key_masks = np.reshape(masks[::self.n_repeat], [-1])
        key_mask_count = len(key_masks)
        k_reparam_indices = np.nonzero(1-key_masks)[0]

        mask = self.mask[1].float().cpu().numpy()
        masks = np.reshape(mask, [self.num_heads, d_head])
        value_masks = np.reshape(masks[::self.n_repeat], [-1])
        v_reparam_indices = np.nonzero(1-value_masks)[0]
        value_mask_count = len(v_reparam_indices)

        if query_mask_count == 0 or key_mask_count == 0 or value_mask_count==0:
            return

        l1 = self.get(self.q_name)
        d = l1.weight.size(1)
        l1.weight.data[q_reparam_indices] = 0.001 * torch.rand(query_mask_count,
                                                             d).to(dtype=self.dtype, device=self.device)
        if l1.bias is not None:
            l1.bias.data[q_reparam_indices] = 0.001 * torch.rand(
                query_mask_count * self.num_heads).to(dtype=self.dtype, device=self.device)

        l1 = self.get(self.k_name)
        l1.weight.data[k_reparam_indices] = 0.001 * torch.rand(key_mask_count,
                                                             d).to(dtype=self.dtype, device=self.device)
        if l1.bias is not None:
            l1.bias.data[k_reparam_indices] = 0.001 * torch.rand(
                key_mask_count * n_head).to(dtype=self.dtype, device=self.device)
        self.mask[0, :] = 1.0


        l1 = self.get(self.v_name)
        d = l1.weight.size(1)
        l1.weight.data[v_reparam_indices] = 0.001 * torch.rand(value_mask_count,
                                                                      d).to(dtype=self.dtype, device=self.device)
        if l1.bias is not None:
            l1.bias.data[v_reparam_indices] = 0.001 * torch.rand(
                value_mask_count * n_head).to(dtype=self.dtype, device=self.device)
        l2 = self.get(self.o_name)
        d = l2.weight.size(0)
        l2.weight.data[:, v_reparam_indices] = 0.001 * torch.rand(d,
                                                                value_mask_count).to(dtype=self.dtype,
                                                                                           device=self.device)
        self.mask[1, :] = 1.0

    def merge_qkv_weight(self):
        if self.qkv_name is not None:
            q_proj = self.get(self.q_name)
            k_proj = self.get(self.k_name)
            v_proj = self.get(self.v_name)

            qkv = torch.cat([q_proj.weight.data, k_proj.weight.data, v_proj.weight.data], axis=1)

            if q_proj.bias is not None:
                bias = torch.cat([q_proj.bias.data, k_proj.bias.data, v_proj.bias.data], axis=0)
                self.ill(self.qkv_name, weight=qkv,bias=bias)
            else:
                self.ill(self.qkv_name, weight=qkv)
            self.set(self.q_name, None)
            self.set(self.k_name, None)
            self.set(self.v_name, None)

    def repeat_kv(self, hidden_states):
        if self.n_repeat == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, self.n_repeat, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * self.n_repeat, slen, head_dim)

    def expand_mask(self, mask):
        if self.n_repeat == 1:
            return mask
        masks = torch.repeat_interleave(mask.view(self.num_heads//self.n_repeat, self.head_dim), self.n_repeat, dim=1).view(
            -1)
        return masks

"""
use diverse masks in attention heads
"""
class GptSparseAttn(SparseAttn):
    q_name = 'q_proj'
    k_name = 'k_proj'
    v_name = 'v_proj'
    qkv_name = 'query_key_value'
    o_name = 'dense'

    def __init__(self, layer=None, layer_idx=0):
        super().__init__()
        # self.layer = layer
        self.layer_index = layer_idx

        self.set_dim(layer)

        if self.qkv_name is None:
            self.cp(layer, self.q_name)
            self.cp(layer, self.k_name)
            self.cp(layer, self.v_name)
        else:
            d = self.head_dim*self.num_heads
            ds = self.head_dim*self.num_heads//self.n_repeat
            self.ill(self.q_name, layer.weight.data[:,:d], None if layer.bias is None else layer.bias.data[:d])
            self.ill(self.k_name, layer.weight.data[:,d:d+ds], None if layer.bias is None else layer.bias.data[d:d+ds])
            self.ill(self.v_name, layer.weight.data[:,d+ds:], None if layer.bias is None else layer.bias.data[d+ds:])

        self.cp(layer, self.o_name)

        layer = self.get(self.q_name)

        self.dtype = layer.weight.dtype
        self.device = layer.weight.device
        mask_size = 2*self.hidden_size//self.n_repeat
        self.mask = Parameter(torch.ones(mask_size).to(dtype=self.dtype, device=self.device),
                                  requires_grad=False)
        self.training = True

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        q_proj = self.get(self.q_name)
        k_proj = self.get(self.k_name)
        v_proj = self.get(self.v_name)
        o_proj = self.get(self.o_name)

        query = q_proj(hidden_states)
        key = k_proj(hidden_states)
        value = v_proj(hidden_states)

        if self.training:
            masks = self.expand_mask(self.mask[:self.hidden_size//self.n_repeat])
            query = query * masks
            key = key * self.mask[:self.hidden_size//self.n_repeat]
            value = value * self.mask[self.hidden_size//self.n_repeat:]

        query_shape = query.size()[:-1] + (self.num_heads, self.head_dim)
        kv_shape = key.size()[:-1] + (self.num_heads//self.n_repeat, self.head_dim)
        query = torch.transpose(query.view(*query_shape),1,2)
        key = torch.transpose(key.view(*kv_shape),1,2)
        value = torch.transpose(value.view(*kv_shape),1,2)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            past_key_value = (key, value)
        else:
            past_key_value = None

        # repeat k/v heads if n_kv_heads < n_heads
        key = self.repeat_kv(key)
        value = self.repeat_kv(value)

        attn_weights = torch.matmul(key,  query.transpose(-1, -2)).transpose(-1, -2)/math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_shape = attn_output.size()[:-2] + (self.num_heads * self.head_dim,)
        attn_output = attn_output.view(attn_shape)
        attn_output = o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

    def set_dim(self, layer):
        self.hidden_size = layer.hidden_size
        self.head_dim = layer.head_dim
        self.num_heads = layer.num_heads
        self.n_repeat = 1

    def update_mask(self, zero_count=1, max_zero_count=1024):

        q_proj = self.get(self.q_name)
        k_proj = self.get(self.k_name)
        v_proj = self.get(self.v_name)

        amps = []
        n_head = self.num_heads//self.n_repeat
        d_head = self.head_dim
        amp_q = torch.sum(q_proj.weight.grad * q_proj.weight, dim=1)
        amp_q = torch.sum(torch.reshape(amp_q, [self.n_repeat, self.num_heads//self.n_repeat, self.head_dim]), dim=0)
        amp_k = torch.sum(k_proj.weight.grad * k_proj.weight, dim=1).reshape(self.num_heads//self.n_repeat, self.head_dim)
        amp1 = amp_q + amp_k
        amps.append(torch.reshape(amp1, [n_head, d_head]))
        amp2 = torch.sum(v_proj.weight.grad * v_proj.weight, dim=1)
        amps.append(torch.reshape(amp2, [n_head, d_head]))

        mask_counts = []
        topk_values = []

        sub_zero_count = max(min(zero_count // self.num_heads, max_zero_count // self.num_heads), 1)
        masks = [self.mask[:self.hidden_size//self.n_repeat], self.mask[self.hidden_size//self.n_repeat:]]
        for i, amp in enumerate(amps):
            mask = masks[i]
            amp = torch.sum(amp, dim=0, keepdim=False)
            mask_mat = torch.reshape(mask, [n_head, d_head])
            topk = torch.topk(torch.abs(amp) + mask_mat * 10000, sub_zero_count, dim=1, largest=False)

            indices = torch.reshape(
                topk.indices[:, :, None] + torch.arange(n_head).reshape(-1, 1, 1).to(self.device) * d_head,
                [-1])
            if torch.sum(mask < 1).item() <= max_zero_count:
                mask[indices] *= 0.9
            else:
                mask[mask < 1] *= 0.9
            mask_counts.append(torch.sum(1 - mask).item())
            topk_values.append(torch.mean(topk.values).item())

        max_grads = [torch.max(q_proj.weight.grad).item(),
                     torch.max(k_proj.weight.grad).item(),
                     torch.max(v_proj.weight.grad).item()]
        return mask_counts, topk_values, max_grads

    def eval(self):

        layer = self.get(self.q_name)
        mask = self.mask[:self.hidden_size//self.n_repeat]
        masks = self.expand_mask(mask)
        layer.weight = Parameter(layer.weight[masks == 1.0].contiguous().detach(), requires_grad=False)
        if layer.bias is not None:
            layer.bias = Parameter(layer.bias[masks == 1.0].contiguous().detach(), requires_grad=False)
        layer.out_features = layer.weight.size(0)

        layer = self.get(self.k_name)
        mask = self.mask[:self.hidden_size//self.n_repeat]
        layer.weight = Parameter(layer.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        if layer.bias is not None:
            layer.bias = Parameter(layer.bias[mask == 1.0].contiguous().detach(), requires_grad=False)
        layer.out_features = layer.weight.size(0)

        layer = self.get(self.v_name)
        mask = self.mask[self.hidden_size//self.n_repeat:]
        layer.weight = Parameter(layer.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        if layer.bias is not None:
            layer.bias = Parameter(layer.bias[mask == 1.0].contiguous().detach(), requires_grad=False)
        layer.out_features = layer.weight.size(0)

        mask = self.mask[self.hidden_size//self.n_repeat:]
        l2 = self.get(self.o_name)
        l2.weight = Parameter(l2.weight[:, mask == 1.0].contiguous().detach(), requires_grad=False)
        if l2.bias is not None:
            l2.bias = Parameter(l2.bias.detach(), requires_grad=False)
        l2.in_features = l2.weight.size(1)

        self.mask = Parameter(torch.reshape(self.mask[self.mask == 1.0], [2, -1]), requires_grad=False)

        d = self.get(self.q_name).out_features
        self.hidden_size = d
        self.head_dim = d // self.num_heads
        self.training = False

"""
use same mask in attention heads
"""
class LlamaSparseAttn(SparseAttn):
    q_name = 'q_proj'
    k_name = 'k_proj'
    v_name = 'v_proj'
    qkv_name = None
    o_name = 'o_proj'
    rotary_emb_name = 'rotary_emb'

    def __init__(self, layer=None, layer_idx=0):
        super().__init__()
        # self.layer = layer
        self.layer_index = layer_idx
        self.config = layer.config

        self.set_dim(layer)

        if self.qkv_name is None:
            self.cp(layer, self.q_name)
            self.cp(layer, self.k_name)
            self.cp(layer, self.v_name)
        else:
            d = self.head_dim*self.num_heads
            dt = self.head_dim*self.num_heads//self.n_repeat
            self.ill(self.q_name, layer.weight.data[:,:d], None if layer.bias is None else layer.bias.data[:d])
            self.ill(self.k_name, layer.weight.data[:,d:d+dt], None if layer.bias is None else layer.bias.data[d:d+dt])
            self.ill(self.v_name, layer.weight.data[:,d+dt:], None if layer.bias is None else layer.bias.data[d+dt:])

        self.cp(layer, self.o_name)
        self.cp(layer, self.rotary_emb_name)

        layer = self.get(self.q_name)
        self.dtype = layer.weight.dtype
        self.device = layer.weight.device

        # share qk and independent v
        mask_size = self.head_dim + self.hidden_size//self.n_repeat
        self.mask = Parameter(torch.ones(mask_size).to(dtype=self.dtype, device=self.device),
                                  requires_grad=False)

        self.training = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q_proj = self.get(self.q_name)
        k_proj = self.get(self.k_name)
        v_proj = self.get(self.v_name)
        o_proj = self.get(self.o_name)
        rotary_emb = self.get(self.rotary_emb_name)

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        if self.training:
            masks = self.mask[:self.head_dim].view(1,self.head_dim).expand(self.num_heads,-1).reshape(-1)
            query_states = query_states * masks
            masks = self.mask[:self.head_dim].view(1,self.head_dim).expand(self.num_heads//self.n_repeat,-1).reshape(-1)
            key_states = key_states * masks
            masks = self.mask[self.head_dim:]
            value_states = value_states * masks

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads//self.n_repeat, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads//self.n_repeat, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = self.repeat_kv(key_states)
        value_states = self.repeat_kv(value_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_dim(self, layer):
        self.hidden_size = layer.hidden_size
        self.head_dim = layer.head_dim
        self.num_heads = layer.num_heads
        self.max_position_embeddings = layer.max_position_embeddings
        self.rope_theta = layer.rope_theta
        self.n_repeat = layer.num_heads//self.config.num_key_value_heads

    def update_mask(self, zero_count=1, max_zero_count=1024):

        q_proj = self.get(self.q_name)
        k_proj = self.get(self.k_name)
        v_proj = self.get(self.v_name)

        amps = []
        n_head = self.num_heads//self.n_repeat
        d_head = self.head_dim
        amp_q = torch.sum(q_proj.weight.grad * q_proj.weight, dim=1)
        amp_q = torch.sum(torch.reshape(amp_q, [self.n_repeat, self.num_heads//self.n_repeat, self.head_dim]), dim=0)
        amp_k = torch.sum(k_proj.weight.grad * k_proj.weight, dim=1)
        amp_k = amp_k.reshape(self.num_heads//self.n_repeat, self.head_dim)
        amp1 = amp_q + amp_k
        amps.append(torch.reshape(amp1, [n_head, d_head]))
        amp2 = torch.sum(v_proj.weight.grad * v_proj.weight, dim=1)
        amps.append(torch.reshape(amp2, [n_head, d_head]))

        mask_counts = []
        topk_values = []

        sub_zero_count = max(min(zero_count // self.num_heads, max_zero_count // self.num_heads), 1)
        masks = [self.mask[:d_head], self.mask[d_head:].view(n_head,d_head)]
        for i, amp in enumerate(amps):
            mask = masks[i]
            if i == 0:  # qk
                amp = torch.sum(amp, dim=0, keepdim=False)
                topk = torch.topk(torch.abs(amp) + mask * 10000, sub_zero_count, dim=0, largest=False)
                indices = topk.indices
                if torch.sum(mask < 1).item()*self.num_heads <= max_zero_count:
                    self.mask[indices] *= 0.9
                else:
                    m = self.mask[:self.head_dim]
                    m[m < 1] *= 0.9
                mask_counts.append(torch.sum(1 - mask).item()*self.num_heads)
            else:  # v
                topk = torch.topk(torch.abs(amp) + mask * 10000, sub_zero_count, dim=1, largest=False)
                indices = torch.reshape(topk.indices[:,:,None] + torch.arange(n_head).reshape(-1, 1, 1).to(self.device) * d_head,
                                    [-1])
                if torch.sum(mask < 1).item()*self.n_repeat <= max_zero_count:
                    self.mask[indices + self.head_dim] *= 0.9
                else:
                    m = self.mask[self.head_dim:]
                    m[m < 1] *= 0.9

                mask_counts.append(torch.sum(1 - mask).item()*self.n_repeat)
            topk_values.append(torch.mean(topk.values).item())

        max_grads = [torch.max(q_proj.weight.grad).item(),
                     torch.max(k_proj.weight.grad).item(),
                     torch.max(v_proj.weight.grad).item()]
        return mask_counts, topk_values, max_grads

    def eval(self):

        l1 = self.get(self.q_name)
        mask = self.mask[:self.head_dim]
        mask = torch.cat([mask] * self.num_heads, dim=0)
        l1.weight = Parameter(l1.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        if l1.bias is not None:
            l1.bias = Parameter(l1.bias[mask == 1.0].contiguous().detach(), requires_grad=False)
        l1.out_features = l1.weight.size(0)

        l1 = self.get(self.k_name)
        mask = self.mask[:self.head_dim]
        mask = torch.cat([mask] * self.n_repeat, dim=0)
        l1.weight = Parameter(l1.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        if l1.bias is not None:
            l1.bias = Parameter(l1.bias[mask == 1.0].contiguous().detach(), requires_grad=False)
        l1.out_features = l1.weight.size(0)

        l1 = self.get(self.v_name)
        mask = self.mask[self.head_dim:]
        mask = self.expand_mask(mask)
        l1.weight = Parameter(l1.weight[mask == 1.0].contiguous().detach(), requires_grad=False)
        if l1.bias is not None:
            l1.bias = Parameter(l1.bias[mask == 1.0].contiguous().detach(), requires_grad=False)
        l1.out_features = l1.weight.size(0)

        l2 = self.get(self.o_name)
        l2.weight = Parameter(l2.weight[:, mask == 1.0].contiguous().detach(), requires_grad=False)
        if l2.bias is not None:
            l2.bias = Parameter(l2.bias.detach(), requires_grad=False)
        l2.in_features = l2.weight.size(1)

        rotary_emb = self.get(self.rotary_emb_name)
        rope_mask = self.mask[:self.head_dim] == 1.0
        rotary_emb.cached_cos = rotary_emb.cached_cos[..., rope_mask]
        rotary_emb.cached_sin = rotary_emb.cached_sin[..., rope_mask]
        self.mask = Parameter(self.mask[self.mask == 1.0], requires_grad=False)

        d = self.get(self.q_name).out_features
        self.hidden_size = d
        self.head_dim = d // self.num_heads
        self.training = False



class SparseDim(SparseModule):
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'
    ln_type = 'ln'

    def __init__(self, layer=None, layer_index=0, mask=None):
        super().__init__()


class GptSparseDim(SparseDim):
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'
    ln_type = 'ln'

    def __init__(self, layer=None, layer_idx=0, mask=None):
        super().__init__()
        self.layer_index = layer_idx
        self.mask = mask

        self.cp(layer, self.attn_name)
        self.cp(layer, self.mlp_name)

        if self.ln_type == 'rms':
            self.set(self.ln1_name, SparseRMSNorm(getattr(layer, self.ln1_name), mask))
            self.set(self.ln2_name, SparseRMSNorm(getattr(layer, self.ln2_name), mask))
        else:
            self.set(self.ln1_name, SparseLayerNorm(getattr(layer, self.ln1_name), mask))
            self.set(self.ln2_name, SparseLayerNorm(getattr(layer, self.ln2_name), mask))

        layer = self.get(self.ln1_name)
        self.dtype = layer.weight.dtype
        self.device = layer.weight.device

        self.training = True

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        mlp = self.get(self.mlp_name)
        attn = self.get(self.attn_name)
        ln1 = self.get(self.ln1_name)
        ln2 = self.get(self.ln2_name)

        residual = hidden_states

        hidden_states = ln1(hidden_states)

        if self.training:
            hidden_states = hidden_states * self.mask

        attn_outputs = attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states

        hidden_states = ln2(hidden_states)

        if self.training:
            hidden_states = hidden_states * self.mask

        feed_forward_hidden_states = mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        # if self.training:
        #     hidden_states = hidden_states*self.mask

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

    def calc_sensitive(self):
        mlp_amp, mlp_max_grads = self.get(self.mlp_name).calc_input_sensitive()
        attn_amp, attn_max_grads = self.get(self.attn_name).calc_input_sensitive()

        return mlp_amp+attn_amp, mlp_max_grads+attn_max_grads

    def eval(self):
        mask_value = self.mask.data
        mask = mask_value == 1.0
        eval_mask = mask_value[mask]

        # MLP
        mlp = self.get(self.mlp_name)
        mlp.eval_input(mask)

        # ATTN
        attn = self.get(self.attn_name)
        attn.eval_input(mask)

        ln1 = self.get(self.ln1_name)
        ln1.eval_input(mask)

        ln2 = self.get(self.ln2_name)
        ln2.eval_input(mask)

        # mask must be set from outer scope, or it will be independent
        self.training = False

    def reparam(self, size=256):
        mask = self.mask.data

        mlp = self.get(self.mlp_name)
        mlp.reparam_input(mask)

        attn = self.get(self.attn_name)
        attn.reparam_input(mask)

        mask = mask.float().cpu().numpy()
        mask_count = int(np.sum(1 - mask))
        reparam_count = min(size, mask_count)
        reparam_indices = np.nonzero(mask)[0][:reparam_count]

        self.mask[reparam_indices] = 1.0


class LlamaSparseDim(SparseDim):
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'
    ln_type = 'rms'

    def __init__(self, layer=None, layer_idx=0, mask=None):
        super().__init__()
        self.layer_index = layer_idx
        self.mask = mask

        self.cp(layer, self.attn_name)
        self.cp(layer, self.mlp_name)

        if self.ln_type == 'rms':
            self.set(self.ln1_name, SparseRMSNorm(getattr(layer, self.ln1_name), mask))
            self.set(self.ln2_name, SparseRMSNorm(getattr(layer, self.ln2_name), mask))
        else:
            self.set(self.ln1_name, SparseLayerNorm(getattr(layer, self.ln1_name), mask))
            self.set(self.ln2_name, SparseLayerNorm(getattr(layer, self.ln2_name), mask))

        layer = self.get(self.ln1_name)
        self.dtype = layer.weight.dtype
        self.device = layer.weight.device

        self.training = True

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True
    ):

        residual = hidden_states

        ln1 = self.get(self.ln1_name)
        ln2 = self.get(self.ln2_name)
        attn = self.get(self.attn_name)
        mlp = self.get(self.mlp_name)

        hidden_states = ln1(hidden_states)

        if self.training:
            hidden_states = hidden_states * self.mask

        hidden_states, self_attn_weights, present_key_value = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        # residual connection
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = ln2(hidden_states)

        if self.training:
            hidden_states = hidden_states * self.mask

        feed_forward_hidden_states = mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        # if self.training:
        #     hidden_states = hidden_states*self.mask

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def calc_sensitive(self):

        mlp = self.get(self.mlp_name)
        attn = self.get(self.attn_name)

        mlp_amp, mlp_max_grads = mlp.calc_input_sensitive()

        attn_amp, attn_max_grads = attn.calc_input_sensitive()

        return mlp_amp+attn_amp, mlp_max_grads+attn_max_grads

    def eval(self):
        mask_value = self.mask.data
        mask = mask_value == 1.0
        eval_mask = mask_value[mask]

        mlp = self.get(self.mlp_name)
        attn = self.get(self.attn_name)
        ln1 = self.get(self.ln1_name)
        ln2 = self.get(self.ln2_name)

        mlp.eval_input(mask)
        #
        # l1 = getattr(mlp, mlp.up_name)
        # l1.weight.requires_grad = False
        # l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        # l1.in_features = l1.weight.size(1)
        #
        # l1 = getattr(mlp, mlp.gate_name)
        # l1.weight.requires_grad = False
        # l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        # l1.in_features = l1.weight.size(1)
        #
        # l2 = getattr(mlp, mlp.down_name)
        # l2.weight.requires_grad = False
        # l2.weight = Parameter(l2.weight[mask].contiguous().detach(), requires_grad=False)
        # if l2.bias is not None:
        #     l2.bias.requires_grad = False
        #     l2.bias = Parameter(l2.bias[mask].detach(), requires_grad=False)
        # l2.out_features = l1.in_features

        attn.eval_input(mask)
        # l1 = getattr(attn, attn.q_name)
        # l1.weight.requires_grad = False
        # l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        # l1.in_features = l1.weight.size(1)
        #
        # l1 = getattr(attn, attn.k_name)
        # l1.weight.requires_grad = False
        # l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        # l1.in_features = l1.weight.size(1)
        #
        # l1 = getattr(attn, attn.v_name)
        # l1.weight.requires_grad = False
        # l1.weight = Parameter(l1.weight[:, mask].contiguous().detach(), requires_grad=False)
        # l1.in_features = l1.weight.size(1)
        #
        # l2 = getattr(attn, attn.o_name)
        # l2.weight.requires_grad = False
        # l2.weight = Parameter(l2.weight[mask].contiguous().detach(), requires_grad=False)
        # if l2.bias is not None:
        #     l2.bias.requires_grad = False
        #     l2.bias = Parameter(l2.bias[mask].detach(), requires_grad=False)
        # l2.out_features = l1.in_features

        ln1.eval_input()

        # ln1.weight = Parameter(ln1.weight[mask].contiguous().detach(),
        #                                         requires_grad=False)
        # ln1.mask = Parameter(eval_mask, requires_grad=False)

        ln2.eval_input()
        # ln2.weight = Parameter(
        #     ln2.weight[mask].contiguous().detach(), requires_grad=False)
        # ln2.mask = Parameter(eval_mask, requires_grad=False)

        # mask must be set from outer scope, or it will be independent
        self.training = False

    def reparam(self, size=256):
        mask = self.mask.data

        mlp = self.get(self.mlp_name)
        mlp.reparam_input(mask)

        attn = self.get(self.attn_name)
        attn.reparam_input(mask)

        mask = mask.float().cpu().numpy()
        mask_count = int(np.sum(1 - mask))
        reparam_count = min(size, mask_count)
        reparam_indices = np.nonzero(mask)[0][:reparam_count]

        self.mask[reparam_indices] = 1.0


class SparseBlock(SparseModule):
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'

    def __init__(self, layer=None, layer_index=0):
        super().__init__()


class LlamaSparseBlock(SparseBlock):
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'

    def __init__(self, layer=None, layer_idx=0):
        super().__init__()
        self.layer_index = layer_idx

        self.cp(layer, self.attn_name)
        self.cp(layer, self.mlp_name)
        self.cp(layer, self.ln1_name)
        self.cp(layer, self.ln2_name)

        layer = self.get(self.ln1_name)
        self.dtype = layer.weight.dtype
        self.device = layer.weight.device

        self.mask = Parameter(torch.ones(()).to(dtype=self.dtype, device=self.device), requires_grad=False)
        self.training = True

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True
    ):

        ln1 = self.get(self.ln1_name)
        ln2 = self.get(self.ln2_name)
        attn = self.get(self.attn_name)
        mlp = self.get(self.mlp_name)

        residual = hidden_states

        hidden_states = ln1(hidden_states)

        hidden_states, self_attn_weights, present_key_value = attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        # residual connection
        if self.training:
            hidden_states = hidden_states * self.mask + residual
        else:
            hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = ln2(hidden_states)

        feed_forward_hidden_states = mlp(hidden_states)

        # residual connection
        if self.training:
            hidden_states = feed_forward_hidden_states * self.mask + residual
        else:
            hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def update_mask(self, zero_count=0.0):
        self.mask.data = torch.tensor(1.0-zero_count).to(dtype=self.dtype,device=self.device)



