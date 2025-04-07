# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import torch
import flood_cuda


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d,))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    flood_cuda.silu_and_mul(out, x)
    return out