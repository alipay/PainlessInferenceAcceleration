# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import flood_cuda
import torch

from flood.utils.benchmark import benchmark_func


def torch_rms_norm(x, w):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + 1e-6)
    return w * x.to(input_dtype)


def flood_rms_norm(x, w):
    flood_cuda.layernorm_forward(x, w, x, 1e-6)
    return x


batch_size = 160
hidden_size = 8192
dtype = torch.bfloat16
# dtype = torch.float16

x = torch.rand(batch_size, hidden_size, dtype=dtype, device="cuda:0")
w = torch.rand(hidden_size, dtype=dtype, device="cuda:0")

n_repeat = 1000
benchmark_func(torch_rms_norm, x, w, n_repeat=n_repeat)
benchmark_func(flood_rms_norm, x, w, n_repeat=n_repeat)
