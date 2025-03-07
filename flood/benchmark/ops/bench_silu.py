# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import flood_cuda
from vllm import _custom_ops as vllm_ops

from flood.utils.benchmark import benchmark_func

batch_size = 1
hidden_size = 2 * 34048
dtype = torch.bfloat16

x = torch.randn(batch_size, hidden_size).to(0).to(dtype)


def vllm_silu(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d,))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    vllm_ops.silu_and_mul(out, x)
    return out


def flood_silu(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d,))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    flood_cuda.silu_and_mul(out, x)
    return out


n_repeat = 1000
benchmark_func(vllm_silu, x, n_repeat=n_repeat)
benchmark_func(flood_silu, x, n_repeat=n_repeat)
