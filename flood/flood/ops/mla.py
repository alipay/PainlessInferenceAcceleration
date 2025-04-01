# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl


"""
deepseek R1 MLA kernel
head_num: 128

q_lora: 1536
q_nope: 128
q_rope: 64

kv_lora: 512
kv_nope: 128
kv_rope: 64

"""






@triton.jit
def batch_causal_mla_kernel(
        Q,
        K,
        V,
        Out,
        softmax_scale,
        stride_q,
        stride_k,
        stride_o,
        q_offsets,
        k_offsets,
        q_lengths,
        k_lengths,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        HEAD: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.program_id(2)
    seqlen_q = tl.load(q_lengths + bid)
    TOKEN = BLOCK_M // GROUP
    if mid * TOKEN >= seqlen_q:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)
    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid)
    seqlen_k = tl.load(k_lengths + bid)

    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H
    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )
    k_ptrs = (
            K + k_offset * stride_k + hid * HEADDIM + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
            V + k_offset * stride_k + hid * HEADDIM + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs,
                    mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
                    other=0.0)

    for n in range(0, seqlen_k, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)

        if EVEN_N:
            k = tl.load(k_ptrs + n * stride_k)
        else:
            k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if not EVEN_N:
            qk += tl.where((n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        if EVEN_N:
            v = tl.load(v_ptrs + n * stride_k)
        else:
            v = tl.load(v_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

    acc_o = acc_o / lse_i[:, None]

    H = stride_o // HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H

    out_ptrs = (
            Out
            + q_offset * stride_o + mid * TOKEN * stride_o + hid * HEADDIM * GROUP + (
                        offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o,
                 mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off)



def batch_mla_fwd(q, k, v, meta, causal=False):
    _, qo_heads, d = q.shape
    _, kv_heads, _ = k.shape
    batch = meta.batch_size
    softmax_scale = 1.0 / math.sqrt(d)

    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    HEADDIM = d

    GROUP = qo_heads // kv_heads
    CAUSAL = causal if meta.max_q_length > 1 else False
    SINGLE = meta.max_seg == 1
    HEAD = qo_heads

    BLOCK_M = 16 if meta.max_q_length == 1 else 128
    BLOCK_N = 128

    EVEN_M = all(
        [x // BLOCK_M * BLOCK_M == x and BLOCK_M // GROUP * GROUP == BLOCK_M for
         x in meta.qls])
    if isinstance(meta.kls[0], (list, tuple)):
        EVEN_N = all(
            [all([y // BLOCK_N * BLOCK_N == y for y in x]) for x in meta.kls])
    else:
        EVEN_N = all([x // BLOCK_N * BLOCK_N == x for x in meta.kls])

    # _CudaDeviceProperties(name='NVIDIA H20', major=9, minor=0, total_memory=97285MB, multi_processor_count=78)
    dp = torch.cuda.get_device_properties(0)
    sm = dp.major * 10 + dp.minor

    TOKEN = BLOCK_M // GROUP
    num_m_block = (meta.max_q_length - 1) // TOKEN + 1
    num_warps = 8
    num_stages = 2 if 80 < sm <= 89 or meta.mask is not None else 3
    grid = lambda META: (batch, kv_heads, num_m_block)


    batch_causal_mla_kernel[grid](
        q,
        k,
        v,
        o,
        softmax_scale,
        q.stride(0),
        k.stride(0),
        o.stride(0),
        meta.q_offsets,
        meta.k_offsets,
        meta.q_lengths,
        meta.k_lengths,
        HEADDIM,
        GROUP,
        HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o


@triton.jit
def single_seg_causal_mla_kernel(
        Q,
        K,
        V,
        Out,
        softmax_scale,
        stride_q,
        stride_k,
        stride_o,
        q_offsets,
        k_offsets,
        q_lengths,
        k_lengths,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        HEAD: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.program_id(2)
    seqlen_q = tl.load(q_lengths + bid)
    TOKEN = BLOCK_M // GROUP
    if mid * TOKEN >= seqlen_q:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)
    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid)
    seqlen_k = tl.load(k_lengths + bid)

    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H
    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )
    k_ptrs = (
            K + k_offset * stride_k + hid * HEADDIM + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
            V + k_offset * stride_k + hid * HEADDIM + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs,
                    mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
                    other=0.0)

    for n in range(0, seqlen_k, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)

        if EVEN_N:
            k = tl.load(k_ptrs + n * stride_k)
        else:
            k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if not EVEN_N:
            qk += tl.where((n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        if EVEN_N:
            v = tl.load(v_ptrs + n * stride_k)
        else:
            v = tl.load(v_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

    acc_o = acc_o / lse_i[:, None]

    H = stride_o // HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H

    out_ptrs = (
            Out
            + q_offset * stride_o + mid * TOKEN * stride_o + hid * HEADDIM * GROUP + (
                        offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o,
                 mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off)



def seg_mla_fwd(q, k, v, meta, causal=False):
    _, qo_heads, d = q.shape
    _, kv_heads, _ = k.shape
    batch = meta.batch_size
    softmax_scale = 1.0 / math.sqrt(d)

    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    HEADDIM = d

    GROUP = qo_heads // kv_heads
    CAUSAL = causal if meta.max_q_length > 1 else False
    SINGLE = meta.max_seg == 1
    HEAD = qo_heads

    BLOCK_M = 16 if meta.max_q_length == 1 else 128
    BLOCK_N = 128

    EVEN_M = all(
        [x // BLOCK_M * BLOCK_M == x and BLOCK_M // GROUP * GROUP == BLOCK_M for
         x in meta.qls])
    if isinstance(meta.kls[0], (list, tuple)):
        EVEN_N = all(
            [all([y // BLOCK_N * BLOCK_N == y for y in x]) for x in meta.kls])
    else:
        EVEN_N = all([x // BLOCK_N * BLOCK_N == x for x in meta.kls])

    # _CudaDeviceProperties(name='NVIDIA H20', major=9, minor=0, total_memory=97285MB, multi_processor_count=78)
    dp = torch.cuda.get_device_properties(0)
    sm = dp.major * 10 + dp.minor

    TOKEN = BLOCK_M // GROUP
    num_m_block = (meta.max_q_length - 1) // TOKEN + 1
    num_warps = 8
    num_stages = 2 if 80 < sm <= 89 or meta.mask is not None else 3
    grid = lambda META: (batch, kv_heads, num_m_block)


    single_seg_causal_mla_kernel[grid](
        q,
        k,
        v,
        o,
        softmax_scale,
        q.stride(0),
        k.stride(0),
        o.stride(0),
        meta.q_offsets,
        meta.k_offsets,
        meta.q_lengths,
        meta.k_lengths,
        HEADDIM,
        GROUP,
        HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o