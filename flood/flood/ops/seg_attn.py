# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import math

import torch
import triton
import triton.language as tl


@triton.jit
def safe_scale(qk, acc_o, lse, maxs, softmax_scale):
    qk *= softmax_scale
    m_i = tl.maximum(maxs, tl.max(qk, 1))
    qk -= m_i[:, None]
    p = tl.exp(qk)
    
    l_i = tl.sum(p, 1)
    alpha = tl.exp(maxs - m_i)
    lse = lse * alpha + l_i
    acc_o = acc_o * alpha[:, None]
    maxs = m_i
    return p, acc_o, lse, maxs

@triton.jit
def fast_scale(qk, softmax_scale, lse):
    p = tl.exp(qk * softmax_scale)
    lse += tl.sum(p, 1)
    return p, lse

@triton.jit
def single_seg_attn_kernel(
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
        MASK: tl.constexpr,
        MASK_SIZE: tl.constexpr,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        TOKEN: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ONLINE_SCALE: tl.constexpr,
        MASK_TYPE: tl.constexpr, # 0: full mask, 1:causal mask, 2:fast customized mask, 3: compatible customized mask
):  
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1

    q_length = tl.load(q_lengths + bid)

    if mid * TOKEN >= q_length:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)
    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid).to(tl.int64)
    k_length = tl.load(k_lengths + bid)

    gap = k_length - q_length
    q_idx = offs_m // GROUP
    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    
    max_m_idx = min(q_length * H, (mid + 1) * TOKEN * H)

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

    lse = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)
    maxs = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs,
                    mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_idx,
                    other=0.0)
    
    if MASK_TYPE == 0:
        for n in range(0, k_length, BLOCK_N):
            n = tl.multiple_of(n, BLOCK_N)

            if EVEN_N:
                k = tl.load(k_ptrs + n * stride_k)
            else:
                k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_length, 
                            other=0.0)
            qk = tl.dot(q, tl.trans(k))
            
            if not EVEN_N:
                qk += tl.where((n + offs_n)[None, :] < k_length, 0, float("-inf"))

            if ONLINE_SCALE:
                p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
            else:
                p, lse = fast_scale(qk, softmax_scale, lse)

            if EVEN_N:
                v = tl.load(v_ptrs + n * stride_k)
            else:
                v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_length, 
                            other=0.0)
            p = p.to(v.dtype)
            acc_o = tl.dot(p, v, acc_o)

    else:
        if MASK_TYPE == 1:
            mask_free_step = (mid * TOKEN + gap) // BLOCK_N
        else:
            # mask size must be equal to q_length
            mask_free_step = gap // BLOCK_N
        
        for i in range(0, mask_free_step):
            n = i * BLOCK_N
            k = tl.load(k_ptrs + n * stride_k)
            qk = tl.dot(q, tl.trans(k))

            if ONLINE_SCALE:
                p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
            else:
                p, lse = fast_scale(qk, softmax_scale, lse)

            v = tl.load(v_ptrs + n * stride_k)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

        q_pos = mid * TOKEN + gap + q_idx[:, None]
        if MASK_TYPE == 2 or MASK_TYPE == 3:
            sub_gap = gap - mask_free_step * BLOCK_N
            if MASK_TYPE == 2:
                mask_ptrs = MASK + bid * MASK_SIZE * MASK_SIZE + \
                    mid * TOKEN * MASK_SIZE - sub_gap + \
                        (tl.arange(0,TOKEN))[:,None] * MASK_SIZE + offs_n[None,:]
            else:
                mask_ptrs = MASK + bid * MASK_SIZE * MASK_SIZE + \
                        mid * TOKEN * MASK_SIZE - sub_gap + \
                            (tl.arange(0,BLOCK_M)%(GROUP*TOKEN)//GROUP)[:,None] * MASK_SIZE + offs_n[None,:]

        for n in range(mask_free_step * BLOCK_N, min((mid + 1) * TOKEN + gap, k_length), BLOCK_N):
            n = tl.multiple_of(n, BLOCK_N)

            if EVEN_N:
                k = tl.load(k_ptrs + n * stride_k)
            else:
                k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_length, 
                            other=0.0)

            qk = tl.dot(q, tl.trans(k))
            if not EVEN_N:
                qk += tl.where((n + offs_n)[None, :] < k_length, 0, float("-inf"))
            
            qk += tl.where(q_pos >= (n + offs_n)[None, :], 0, float("-inf"))
            
            if MASK_TYPE == 2:
                qk = tl.reshape(qk, (TOKEN, GROUP, BLOCK_N), can_reorder=False)
                mask = tl.load(mask_ptrs, mask=(offs_n[None, :] < MASK_SIZE) & (
                        offs_n[None, :] >= sub_gap), other=0.0)
                mask = -10000 * mask.to(tl.float32)
                qk += mask[:, None, :]
                qk = tl.reshape(qk, (TOKEN * GROUP, BLOCK_N), can_reorder=False)
            elif MASK_TYPE == 3:
                mask = tl.load(mask_ptrs, mask=(offs_n[None, :] < MASK_SIZE) & \
                        (offs_n[None, :] >= sub_gap), other=0.0)
                qk = tl.where(mask==0, qk, qk-10000.0)

            if ONLINE_SCALE:
                p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
            else:
                p, lse = fast_scale(qk, softmax_scale, lse)

            if EVEN_N:
                v = tl.load(v_ptrs + n * stride_k)
            else:
                v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_length, 
                            other=0.0)

            p = p.to(v.dtype)

            acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    H = stride_o // HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_idx = min(q_length, (mid + 1) * TOKEN) * H

    out_ptrs = (
            Out
            + q_offset * stride_o + mid * TOKEN * stride_o + hid * HEADDIM * GROUP + (
                    offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o,
                 mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_idx)


@triton.jit
def multi_seg_attn_kernel(
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
        k_segs,
        max_seg,
        MASK: tl.constexpr,
        MASK_SIZE: tl.constexpr,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        TOKEN: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ONLINE_SCALE: tl.constexpr,
        MASK_TYPE: tl.constexpr, # 0: full mask, 1:causal, 2:customized mask, 3: compatible
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1

    q_length = tl.load(q_lengths + bid)
    TOKEN = BLOCK_M // GROUP

    if mid * TOKEN >= q_length:
        return
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)
    q_offset = tl.load(q_offsets + bid)
    n_seg = tl.load(k_segs + bid)
    seqlen_total_k = tl.load(k_lengths + bid * (max_seg + 1) + max_seg)

    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(q_length * H, (mid + 1) * TOKEN * H)
    
    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    lse = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)
    maxs = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    q = tl.load(q_ptrs, mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
            other=0.0)
    gap = seqlen_total_k - q_length

    if MASK_TYPE == 0:
        for i_seg in range(n_seg):
            k_offset = tl.load(k_offsets + bid * max_seg + i_seg).to(tl.int64)
            k_length = tl.load(k_lengths + bid * (max_seg + 1) + i_seg)

            k_ptrs = (
                    K + k_offset * stride_k + hid * HEADDIM + (
                        offs_n[:, None] * stride_k + offs_d[None, :])
            )
            v_ptrs = (
                    V + k_offset * stride_k + hid * HEADDIM + (
                        offs_n[:, None] * stride_k + offs_d[None, :])
            )

            for n in range(0, k_length, BLOCK_N):
                n = tl.multiple_of(n, BLOCK_N)
                k = tl.load(k_ptrs + n * stride_k, mask=(n + offs_n)[:, None] < k_length, other=0.0)
                qk = tl.dot(q, tl.trans(k))
                if not EVEN_N:
                    qk += tl.where((n + offs_n)[None, :] < k_length, 0,
                            float("-inf"))

                if ONLINE_SCALE:
                    p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
                else:
                    p, lse = fast_scale(qk, softmax_scale, lse)

                v = tl.load(v_ptrs + n * stride_k, mask=(n + offs_n)[:, None] < k_length, other=0.0)
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)
    else:
        for i_seg in range(n_seg - 1):
            k_offset = tl.load(k_offsets + bid * max_seg + i_seg).to(tl.int64)
            k_seg_length = tl.load(k_lengths + bid * (max_seg + 1) + i_seg)

            k_ptrs = (
                    (K + k_offset * stride_k + hid * HEADDIM) + (
                        offs_n[:, None] * stride_k + offs_d[None, :])
            )
            v_ptrs = (
                    (V + k_offset * stride_k + hid * HEADDIM) + (
                        offs_n[:, None] * stride_k + offs_d[None, :])
            )

            for i in range((k_seg_length - 1) // BLOCK_N + 1):
                n = i * BLOCK_N
                n = tl.multiple_of(n, BLOCK_N)
                if EVEN_N:
                    k = tl.load(k_ptrs + n * stride_k)
                else:
                    k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_seg_length,
                                other=0.0)
                qk = tl.dot(q, tl.trans(k))

                if not EVEN_N:
                    qk += tl.where((n + offs_n)[None, :] < k_seg_length, 0, float("-inf"))

                if ONLINE_SCALE:
                    p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
                else:
                    p, lse = fast_scale(qk, softmax_scale, lse)

                if EVEN_N:
                    v = tl.load(v_ptrs + n * stride_k)
                else:
                    v = tl.load(v_ptrs + n * stride_k,
                                mask=(n + offs_n)[:, None] < k_seg_length,
                                other=0.0)
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)

        # k_length_accum = tl.load(k_lengths + bid * (max_seg + 1) + n_seg - 1)
        # k_length_accum_next = tl.load(k_lengths + bid * (max_seg + 1) + n_seg)
        k_offset = tl.load(k_offsets + bid * max_seg + n_seg - 1).to(tl.int64)
        k_seg_length = tl.load(k_lengths + bid * (max_seg + 1) + n_seg - 1)
        k_length_accum = seqlen_total_k - k_seg_length

        k_ptrs = (
                (K + k_offset * stride_k + hid * HEADDIM) + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )
        v_ptrs = (
                (V + k_offset * stride_k + hid * HEADDIM) + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )

        mask_free_step = min(max(mid * TOKEN + gap - k_length_accum, 0),
                k_seg_length) // BLOCK_N
        
        for i in range(mask_free_step):
            n = i * BLOCK_N
            n = tl.multiple_of(n, BLOCK_N)
            k = tl.load(k_ptrs + n * stride_k)
            qk = tl.dot(q, tl.trans(k))

            if ONLINE_SCALE:
                p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
            else:
                p, lse = fast_scale(qk, softmax_scale, lse)

            v = tl.load(v_ptrs + n * stride_k)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

        q_pos = ((mid * TOKEN + gap) + tl.arange(0, BLOCK_M) // GROUP)[:, None]
        max_k_idx = min(mid * TOKEN + TOKEN + gap - k_length_accum, k_seg_length)

        if MASK_TYPE == 2:
            sub_gap = gap - k_length_accum - mask_free_step * BLOCK_N
            mask_ptrs = MASK + bid * MASK_SIZE * MASK_SIZE + mid * TOKEN * MASK_SIZE - sub_gap + \
                    (tl.arange(0, TOKEN))[:, None] * MASK_SIZE + offs_n[None, :]

        for n in range(mask_free_step * BLOCK_N, max_k_idx, BLOCK_N):
            n = tl.multiple_of(n, BLOCK_N)

            if MASK_TYPE == 1:
                if EVEN_N:
                    k = tl.load(k_ptrs + n * stride_k)
                else:
                    k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_seg_length, other=0.0)
                qk = tl.dot(q, tl.trans(k))
                
                if not EVEN_N:
                    qk += tl.where((n + offs_n)[None, :] < k_seg_length, 0,
                            float("-inf"))
            else:
                k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < k_seg_length, other=0.0)
                qk = tl.dot(q, tl.trans(k))
                qk += tl.where((n + offs_n)[None, :] < k_seg_length, 0, float("-inf"))


            qk += tl.where(q_pos >= (k_length_accum + n + offs_n)[None, :], 0,
                    float("-inf"))

            if MASK_TYPE == 2:
                # mask
                qk = tl.reshape(qk, (TOKEN, GROUP, BLOCK_N), can_reorder=False)
                mask = tl.load(mask_ptrs, mask=(offs_n[None, :] < MASK_SIZE) & (
                        offs_n[None, :] >= sub_gap), other=0.0)
                mask = -10000 * mask.to(tl.float32)
                qk += mask[:, None, :]
                qk = tl.reshape(qk, (TOKEN * GROUP, BLOCK_N), can_reorder=False)

            if ONLINE_SCALE:
                p, acc_o, lse, maxs = safe_scale(qk, acc_o, lse, maxs, softmax_scale)
            else:
                p, lse = fast_scale(qk, softmax_scale, lse)

            if MASK_TYPE == 1:
                if EVEN_N:
                    v = tl.load(v_ptrs + n * stride_k)
                else:
                    v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < k_seg_length, other=0.0)
            else:
                v = tl.load(v_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < k_seg_length, other=0.0)
            
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
    

    acc_o = acc_o / lse[:, None]

    H = stride_o // HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(q_length, (mid + 1) * TOKEN) * H

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


"""
limitation: q length should be shorter than the last kv segment length
TODO: ADAPT FOR THE LIMITATION
"""


def seg_attn_fwd(q, k, v, meta, online_scale=True):
    _, qo_heads, d = q.shape
    _, kv_heads, _ = k.shape
    batch = meta.batch_size
    softmax_scale = 1.0 / math.sqrt(d)

    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    HEADDIM = d

    GROUP = qo_heads // kv_heads
    SINGLE = meta.max_seg == 1
    BLOCK_M = 16 if meta.max_q_length == 1 else 128
    BLOCK_N = 128
    TOKEN = BLOCK_M // GROUP

    EVEN_M = all(
        [x % BLOCK_M == 0 and BLOCK_M % GROUP == 0 for
         x in meta.qls])
    if isinstance(meta.kls[0], (list, tuple)):
        EVEN_N = all(
            [all([y % BLOCK_N == 0 for y in x]) for x in meta.kls])
    else:
        EVEN_N = all([x % BLOCK_N == 0 for x in meta.kls])

    # _CudaDeviceProperties(name='NVIDIA H20', major=9, minor=0, total_memory=97285MB, multi_processor_count=78)
    prop = torch.cuda.get_device_properties(0)
    sm = prop.major * 10 + prop.minor

    if meta.mask is not None:
        MASK_TYPE = 2 if BLOCK_M % GROUP == 0 else 3
    elif meta.max_q_length > 1:
        MASK_TYPE = 1
    else:
        MASK_TYPE = 0

    num_m_block = (meta.max_q_length - 1) // TOKEN + 1
    num_warps = 8  # TODO: only for compatible mode
    if meta.mask is not None and BLOCK_M % GROUP != 0:
        num_stages = 1
    elif meta.mask is not None and BLOCK_M % GROUP == 0:
        num_stages = 2
    else:
        num_stages = 3
    if sm not in (80, 90):
        num_stages = max(1, num_stages - 1)
    grid = lambda META: (batch, kv_heads, num_m_block)
    if SINGLE:
        single_seg_attn_kernel[grid](
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
            MASK=meta.mask,
            MASK_SIZE=0 if meta.mask is None else meta.mask.size(-1),
            HEADDIM=HEADDIM,
            GROUP=GROUP,
            TOKEN=TOKEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            num_warps=num_warps,
            num_stages=num_stages,
            ONLINE_SCALE=online_scale,
            MASK_TYPE=MASK_TYPE
        )
        return o
    else:
        multi_seg_attn_kernel[grid](
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
            meta.k_segs,
            meta.max_seg,
            MASK=meta.mask,
            MASK_SIZE=0 if meta.mask is None else meta.mask.size(-1),
            HEADDIM=HEADDIM,
            GROUP=GROUP,
            TOKEN=None if meta.mask is None else TOKEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            num_warps=num_warps,
            num_stages=num_stages,
            ONLINE_SCALE=online_scale,
            MASK_TYPE=MASK_TYPE
        )
        return o
