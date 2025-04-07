# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl

import flood_cuda


# @triton.jit
# def update_cache_kernel(
#         Out,
#         States,
#         Indices,
#         DIM: tl.constexpr,
#         BLOCK: tl.constexpr
# ):
#     pid = tl.program_id(0)

#     idx = tl.load(Indices + pid)
#     n = DIM//BLOCK
#     indices = tl.arange(0,BLOCK)
#     for i in range(n):
#         v = tl.load(States+pid*DIM+i*BLOCK+indices)
#         tl.store(Out+idx*DIM+i*BLOCK+indices, v)

# def triton_update_fusion_cache(kv_out: torch.Tensor,
#         key_value_states: torch.Tensor,
#         indices: torch.Tensor):
#     # kv_out: [long, dim]
#     # key_value_states: [q_len, dim]
#     n_token, dim = key_value_states.size()

#     BLOCK = max([x for x in [64,128,256,512,1024,2048] if dim%x==0])
#     grid = lambda META: (n_token,)
#     update_cache_kernel[grid](
#         kv_out,
#         key_value_states,
#         indices,
#         DIM=dim,
#         BLOCK=BLOCK,
#         num_warps=4,
#         num_stages=4
#     )
#     return kv_out




def update_cache(
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        indices: torch.Tensor
):
    flood_cuda.update_cache(k_out, v_out,
                            key_states, value_states,
                            indices, 
                            key_states.size(0),
                            key_states.size(1) * key_states.size(2),
                            key_states.stride(0) // 8)


def update_fusion_cache(
        kv_out: torch.Tensor,
        kv_states: torch.Tensor,
        indices: torch.Tensor
):
    flood_cuda.update_fusion_cache(kv_out, kv_states,
                                   indices, 
                                   kv_states.size(0),
                                   kv_states.size(1),
                                   kv_states.stride(0) // 8)


def quant_and_update_cache(
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        indices: torch.Tensor
):
    n_token = key_states.size(0)
    kv_dim = key_states.size(1) * key_states.size(2)
    q_dim = query_states.size(1) * query_states.size(2)
    group = q_dim // kv_dim
    kv_stride = key_states.stride(0) // 8
    q_stride = query_states.stride(0) // 8
    flood_cuda.quant_to_fp8_and_update_cache(q_out, k_out, v_out,
                                             query_states, key_states,
                                             value_states,
                                             indices,
                                             n_token, group, kv_dim, q_stride,
                                             kv_stride)

