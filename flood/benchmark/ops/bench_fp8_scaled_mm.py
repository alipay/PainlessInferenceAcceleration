# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import torch
import flood_cuda
from vllm import _custom_ops as ops

from flood.utils.benchmark import benchmark_func


def torch_scaled_mm(x, weight, x_scale, weight_scale):
    output = torch._scaled_mm(
        x,
        weight,
        scale_a=x_scale,
        scale_b=weight_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    return output


def vllm_scaled_mm(x, weight, x_scale, weight_scale):
    out = ops.cutlass_scaled_mm(
        x,
        weight,
        out_dtype=torch.bfloat16,
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=None,
    )
    return out


for i in range(1, 11):
    batch_size = 2**i
    in_dim = 4096
    out_dim = 8192
    dtype = torch.bfloat16
    x = torch.randn(batch_size, in_dim, dtype=dtype, device="cuda:0").to(
        torch.float8_e4m3fn
    )
    w = (
        torch.randn(out_dim, in_dim, dtype=dtype, device="cuda:0")
        .to(torch.float8_e4m3fn)
        .t()
    )
    x_scale = torch.rand(batch_size, 1, dtype=torch.float32, device="cuda:0")
    weight_scale = torch.rand(out_dim, 1, dtype=torch.float32, device="cuda:0")

    n_repeat = 100
    print("\n")
    benchmark_func(
        torch_scaled_mm,
        x,
        w,
        x_scale,
        weight_scale.t(),
        n_repeat=n_repeat,
        desc=f"bs:{batch_size}",
    )
    benchmark_func(
        vllm_scaled_mm,
        x,
        w,
        x_scale,
        weight_scale,
        n_repeat=n_repeat,
        desc=f"bs:{batch_size}",
    )
