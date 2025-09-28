# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random
import torch
import flood_cuda

from flood.ops.cache import triton_update_fusion_cache, update_fusion_cache
from flood.utils.benchmark import benchmark_func


def torch_update_cache(k_out, v_out, key_states, value_states, indices):
    k_out[indices] = key_states
    v_out[indices] = value_states


def flood_update_cache(k_out, v_out, key_states, value_states, indices):
    flood_cuda.update_cache(
        k_out,
        v_out,
        key_states,
        value_states,
        indices,
        key_states.size(0),
        key_states.size(1) * key_states.size(2),
        key_states.stride(0) // 8,
    )


bs = 1024
offset = 8

key_states = torch.rand(bs, 8 + offset, 128, dtype=torch.bfloat16, device="cuda:0")[
    :, :8
]
value_states = torch.rand(bs, 8 + offset, 128, dtype=torch.bfloat16, device="cuda:0")[
    :, :8
]

torch_k_out = torch.randn(262144, 8, 128, dtype=torch.bfloat16, device="cuda:0")
torch_v_out = torch.randn(262144, 8, 128, dtype=torch.bfloat16, device="cuda:0")

flood_k_out = torch.randn(262144, 8, 128, dtype=torch.bfloat16, device="cuda:0")
flood_v_out = torch.randn(262144, 8, 128, dtype=torch.bfloat16, device="cuda:0")

indices = []
for i in range(bs):
    indices.append(random.randint(0, 2**18 - 1))
indices = torch.tensor(indices, dtype=torch.int32, device="cuda:0")

n_repeat = 1000
benchmark_func(
    torch_update_cache,
    torch_k_out,
    torch_v_out,
    key_states,
    value_states,
    indices,
    n_repeat=n_repeat,
)
benchmark_func(
    flood_update_cache,
    flood_k_out,
    flood_v_out,
    key_states,
    value_states,
    indices,
    n_repeat=n_repeat,
)

benchmark_func(update_fusion_cache, flood_k_out, key_states, indices, n_repeat=n_repeat)

benchmark_func(
    triton_update_fusion_cache,
    flood_k_out.view(-1, 1024),
    key_states.view(-1, 1024),
    indices,
    n_repeat=n_repeat,
)
