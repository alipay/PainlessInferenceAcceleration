# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from torch.autograd import forward_ad
import flood_cuda

import triton
import triton.language as tl

class RMSNorm(torch.nn.Module):

    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size),
                                         requires_grad=False)

    def forward(self, x: torch.Tensor):
        y = torch.empty_like(x)
        flood_cuda.rmsnorm(x, self.weight, y, self.variance_epsilon)
        return y

class RMSGroupNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, num_norm_group: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.num_norm_group = num_norm_group
        self.per_group_hidden_size = hidden_size // num_norm_group

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        x = x.view(-1, self.num_norm_group, self.per_group_hidden_size)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        x = x.view(-1, self.hidden_size)
        return x * self.weight

class RMSGroupNormSigmoid(torch.nn.Module):
    def __init__(self, hidden_size: int, num_norm_group: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.num_norm_group = num_norm_group
        self.per_group_hidden_size = hidden_size // num_norm_group

    def forward(self,x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return triton_rms_groupnorm_sigmoid(x, self.weight, g, self.variance_epsilon, self.per_group_hidden_size)


@triton.jit
def rms_groupnorm_sigmoid_kernel(
    x_ptr,
    weight_ptr, 
    g_ptr,
    out_ptr,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    x_stride: tl.constexpr,
    g_stride: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK: tl.constexpr
):
    bid = tl.program_id(0)
    gid = tl.program_id(1)

    seq_off = bid * BLOCK + tl.arange(0, BLOCK)
    seq_mask = seq_off < M
    g_off = gid * group_size + tl.arange(0, group_size)
    g_mask = g_off < N
    mask = seq_mask[:, None] & g_mask[None, :]

    x = tl.load(x_ptr + seq_off[:, None] * x_stride + g_off[None, :], mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + g_off)
    g = tl.load(g_ptr + seq_off[:, None] * g_stride + g_off[None, :] , mask=mask, other=0.0).to(tl.float32)
    rms = 1/tl.sqrt(tl.sum(x*x, 1) / group_size + eps)
    x *= rms[:, None]
    x = x.to(w.dtype) * w[None, :]
    x *= tl.sigmoid(g).to(w.dtype)
    tl.store(out_ptr + seq_off[:, None] * N + g_off, x, mask=mask)

def triton_rms_groupnorm_sigmoid(x, weight, g, eps, group_size):
    assert x.shape[1] == weight.shape[0]
    assert x.shape[1] % group_size == 0
    dtype = x.dtype
    device = x.device
    M, N = x.shape
    x_stride = x.stride(0)
    g_stride = g.stride(0)

    BLOCK = 32
    num_stages = 2
    num_warps = 8

    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, group_size))
    out = torch.empty_like(x, dtype=dtype, device=device)
    rms_groupnorm_sigmoid_kernel[grid](
        x,
        weight,
        g,
        out,
        eps,
        M,
        N,
        x_stride,
        g_stride,
        group_size,
        BLOCK,
        num_stages=num_stages,
        num_warps=num_warps
        )
    return out

