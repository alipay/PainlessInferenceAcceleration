# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from math import log
import torch
import flood_cuda

from flood.ops.norm import triton_rms_groupnorm_sigmoid

from flood.utils.benchmark import benchmark_func


def torch_rms_norm(x, w, eps=1e-6):
    output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(
        x.dtype)
    return output * w


def flood_rms_norm(x, w, eps=1e-6):
    y = torch.empty_like(x)
    flood_cuda.rmsnorm(x, w, y, eps)
    return y


def torch_group_rms_norm(x, w, eps=1e-6, hidden_size=8192, n_group=4):
    x = x.view(-1, n_group, hidden_size//n_group)
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)
    x = x.view(-1, hidden_size)
    return x * w

def test_group_rms_norm_silu(x, w, g, eps=1e-6, hidden_size=5120, group_size=4):
    output = torch_group_rms_norm(x, w, eps, hidden_size, group_size) * torch.sigmoid(g)
    return output

batch_size = 128
hidden_size = 2048
# dtype = torch.float16
dtype = torch.bfloat16
device = 'cuda:0'
eps = 1e-5

x_l = torch.randn(batch_size, 2 * hidden_size).to(device).to(dtype)
x = torch.split(x_l, [hidden_size, hidden_size] , dim=-1)[0]

w = torch.randn(hidden_size).to(device).to(dtype)

y_ref = torch_rms_norm(x, w, eps=eps)
y_ker = flood_rms_norm(x, w, eps=eps)
torch.testing.assert_close(y_ref, y_ker, rtol=2 / 2 ** 7, atol=1e-2)

x = torch.randn(batch_size, hidden_size).to(device).to(dtype)
y_ker = flood_rms_norm(x, w, eps=eps)
y_ref = torch_rms_norm(x, w, eps=eps)

torch.testing.assert_close(y_ref, y_ker, rtol=2 / 2 ** 7, atol=1e-2)

g = torch.randn(batch_size, hidden_size).to(device).to(dtype)
n_group = 4
group_size = hidden_size // n_group

y_ker = test_group_rms_norm_silu(x, w, g, eps, hidden_size, n_group)
y_ref = triton_rms_groupnorm_sigmoid(x, w, g, eps, group_size)
torch.testing.assert_close(y_ref, y_ker, rtol=2 / 2 ** 7, atol=1e-2)


benchmark_func(test_group_rms_norm_silu, x, w, g, eps, hidden_size, n_group, n_repeat=1000)

benchmark_func(triton_rms_groupnorm_sigmoid, x, w, g, eps, group_size, n_repeat=1000, n_profile=10, trace_dir='./triton_rms_groupnorm_sigmoid.json')