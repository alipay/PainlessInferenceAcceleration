# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F

import flood_cuda

batch_size = 1
hidden_size = 2 * 34048
dtype = torch.bfloat16

x = torch.randn(batch_size, hidden_size).to(0).to(dtype)


def torch_silu(x: torch.Tensor) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def flood_silu(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d,))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    flood_cuda.silu_and_mul(out, x)
    return out


torch_out = torch_silu(x)

flood_out = flood_silu(x)

torch.testing.assert_close(
    torch_out, flood_out, rtol=1e-4, atol=1e-4
)
