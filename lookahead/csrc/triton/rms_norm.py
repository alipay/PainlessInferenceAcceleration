# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


def rmsnorm_torch(x: torch.Tensor, rms_weights: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms_weights


@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,  # 4096000 4096 1
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    # tl.device_print("stride_x_batch: ", stride_x_batch)

    # parallel at m dimention
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)

    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)  # careful stride_x_k
        var += tl.math.pow(x.to(tl.float32), 2)

    # tl.device_print("var: ", var) 
    var = tl.sum(var, axis=0) / N_SIZE  # reduce 
    std = tl.math.rsqrt(var + eps)
    # tl.device_print("var: ", var)

    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE

        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)

        x_new = x * std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)


def rmsnorm_wrapper(x, rms_weights, eps=1e-6):
    batch, M, K = x.shape
    # print(x.shape) #1, 1000, 4096
    # assert rms_weights.shape[-1] == K
    out = torch.empty_like(x)
    # print(x.stride())
    # print(out.stride())
    rmsnorm_triton[(batch, M,)](x, rms_weights, out,
                                *x.stride(),  # 4096000 4096 1
                                *rms_weights.stride(),  # 1
                                *out.stride(),  # 4096000 4096 1
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=4096,
                                num_warps=16
                                )
    return out
