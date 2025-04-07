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
