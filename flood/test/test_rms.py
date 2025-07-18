# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import flood_cuda


def torch_rms_norm(x, w, eps=1e-6):
    output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(
        x.dtype)
    return output * w


def flood_rms_norm(x, w, eps=1e-6):
    y = torch.empty_like(x)
    flood_cuda.rmsnorm(x, w, y, eps)
    return y


batch_size = 16
hidden_size = 8192
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
