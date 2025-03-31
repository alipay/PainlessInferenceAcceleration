

# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl

from flood.utils.benchmark import benchmark_func

@triton.jit
def static_int8_gemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s,
    b_s,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # a token-wise quantization, b channel-wise quantization.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    scale = a_s*b_s

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = (accumulator.to(tl.float32)*scale).to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def static_int8_gemm_nt(a: torch.Tensor, b: torch.Tensor, a_s: float,  b_s: float, dtype: torch.types):
    assert a.is_contiguous() and b.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa: E731
    static_int8_gemm_nt_kernel[grid](a, b, c, a_s, b_s, M, N, K, BLOCK_SIZE_K=128, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128)
    return c


@triton.jit
def dynamic_int8_gemm_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # a token-wise quantization, b channel-wise quantization.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + (pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
    b_s_ptrs = b_s_ptr + (pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    
    a_s = tl.load(a_s_ptrs, mask=pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)<M)
    b_s = tl.load(b_s_ptrs, mask=pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)<N)

    accumulator = accumulator.to(tl.float32) * a_s[:,None] * b_s[None, :]
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def dynamic_int8_gemm_nt(a: torch.Tensor, b: torch.Tensor, a_s: torch.Tensor,  b_s: torch.Tensor, dtype: torch.types):
    assert a.is_contiguous() and b.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=dtype)
    BLOCK_SIZE_M=128
    BLOCK_SIZE_N=128
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))  # noqa: E731
    dynamic_int8_gemm_nt_kernel[grid](a, b, c, a_s, b_s, M, N, K, BLOCK_SIZE_K=128, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128)
    return c



if __name__ == '__main__':


    mode = 'dynamic'
    if mode == 'static':
        M, N, K = 4096, 4096, 8192
        x = (10*torch.randn((M,K),dtype=torch.bfloat16,device='cuda:0')).to(torch.int8)
        w = (10*torch.randn((N,K),dtype=torch.bfloat16,device='cuda:0')).to(torch.int8)
        y = static_int8_gemm_nt(x,w,1.0,1.0,torch.bfloat16)
        ref_y = x.float()@w.float().t()
        error = (y.float()-ref_y).abs().mean().item()
        print(f'error:{error:.3f}')
        benchmark_func(static_int8_gemm_nt, x,w,1.0,1.0,torch.bfloat16, ref_flops=M*N*K*2)
    else:
        M, N, K = 4096, 4096, 8192
        x = (10*torch.randn((M,K),dtype=torch.bfloat16,device='cuda:0')).to(torch.int8)
        w = (10*torch.randn((N,K),dtype=torch.bfloat16,device='cuda:0')).to(torch.int8)
        xs = torch.rand((M,),dtype=torch.float32,device='cuda:0')
        ws = torch.rand((N,),dtype=torch.float32,device='cuda:0')
        y = dynamic_int8_gemm_nt(x,w,xs,ws,torch.bfloat16)
        ref_y = (x.float()*xs[:,None])@(w.float().t()*ws)
        error = (y.float()-ref_y).abs().mean().item()
        print(f'error:{error:.3f}')
        print(f'{y=}')
        print(f'{ref_y=}')
        benchmark_func(dynamic_int8_gemm_nt, x,w,xs,ws,torch.bfloat16, ref_flops=M*N*K*2)