

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



@triton.jit
def fp8_tb_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # a tilewise quantization, b blockwise quantization.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        # a = tl.load(a_ptrs)
        # b = tl.load(b_ptrs)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        # accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        accumulators = tl.dot(a, b, accumulator)
        accumulator += (accumulators-accumulator) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    # tl.store(c_ptrs, c)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_tb_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, dtype: torch.types):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=dtype)
    block_size = K // a_s.size(-1)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa: E731
    fp8_tb_gemm_kernel[grid](a, b, c, a_s, b_s, 
                             M, N, K, 
                             BLOCK_SIZE_M=block_size, 
                             BLOCK_SIZE_N=block_size, 
                             BLOCK_SIZE_K=block_size, 
                             num_warps=8, 
                             num_stages=4)
    return c


