# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import flood_cuda

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
    def __init__(self, hidden_size: int, linear_attn_norm_group_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.linear_attn_norm_group_size = linear_attn_norm_group_size
        self.per_group_hidden_size = hidden_size // linear_attn_norm_group_size

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        x = x.view(-1, self.linear_attn_norm_group_size, self.per_group_hidden_size)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        x = x.view(-1, self.hidden_size)
        return x * self.weight
    
