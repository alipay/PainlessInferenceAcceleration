# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import math

import torch
import triton
import triton.language as tl

"""
used for decode
"""


@triton.jit
def single_seg_full_attn_kernel(
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


@triton.jit
def single_seg_causal_attn_kernel(
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
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    seqlen_q = tl.load(q_lengths + bid)
    TOKEN = BLOCK_M // GROUP
    if mid * TOKEN >= seqlen_q:
        return

    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid)
    seqlen_k = tl.load(k_lengths + bid)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)

    gap = seqlen_k - seqlen_q

    q_idx = offs_m // GROUP
    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H

    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )
    k_ptrs = (
            (K + k_offset * stride_k + hid * HEADDIM) + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
            (V + k_offset * stride_k + hid * HEADDIM) + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)

    # q = tl.load(q_ptrs)
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs,
                    mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
                    other=0.0)

    mask_free_step = (mid * TOKEN + gap) // BLOCK_N
    for i in range(0, mask_free_step):
        n = i * BLOCK_N
        k = tl.load(k_ptrs + n * stride_k)

        qk = tl.dot(q, tl.trans(k))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_k)

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

    q_pos = (mid * TOKEN + gap) + q_idx[:, None]
    for n in range(mask_free_step * BLOCK_N, mid * TOKEN + TOKEN + gap,
                   BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        # k = tl.load(k_ptrs + n * stride_k)
        if EVEN_N:
            k = tl.load(k_ptrs + n * stride_k)
        else:
            k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        qk += tl.where(q_pos >= (n + offs_n)[None, :], 0, float("-inf"))
        # qk += tl.where((mid * BLOCK_M + gap + offs_repeat)[:, None] >= (n + offs_n)[None, :], 0, float("-inf"))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        # v = tl.load(v_ptrs + n * stride_k)
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


@triton.jit
def single_seg_mask_attn_kernel(
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
        Mask,
        MASK_SIZE: tl.constexpr,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        TOKEN: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    seqlen_q = tl.load(q_lengths + bid)
    # TOKEN = BLOCK_M//GROUP  # BLOCK_M must be multiple of GROUP, or else failure with mask
    if mid * TOKEN >= seqlen_q:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)
    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid)
    seqlen_k = tl.load(k_lengths + bid)
    gap = seqlen_k - seqlen_q

    q_idx = offs_m // GROUP
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

    mask_free_step = gap // BLOCK_N
    for i in range(0, mask_free_step):
        n = i * BLOCK_N

        k = tl.load(k_ptrs + n * stride_k)
        qk = tl.dot(q, tl.trans(k))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_k)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

    sub_gap = gap - mask_free_step * BLOCK_N
    mask_ptrs = Mask + bid * MASK_SIZE * MASK_SIZE + \
             mid * TOKEN * MASK_SIZE - sub_gap + \
                 (tl.arange(0,TOKEN))[:,None] * MASK_SIZE + offs_n[None,:]
    q_pos = (mid * TOKEN + gap) + q_idx[:, None]
    for n in range(mask_free_step * BLOCK_N, mid * TOKEN + TOKEN + gap, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        if EVEN_N:
            k = tl.load(k_ptrs + n * stride_k)
        else:
            k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        qk += tl.where(q_pos >= (n + offs_n)[None, :], 0, float("-inf"))

        # mask
        qk = tl.reshape(qk, (TOKEN, GROUP, BLOCK_N), can_reorder=False)
        mask = tl.load(mask_ptrs, mask=(offs_n[None, :] < MASK_SIZE) & (
                    offs_n[None, :] >= sub_gap), other=0.0)
        mask = -10000 * tl.cast(mask, tl.float32)
        qk += mask[:, None, :]
        qk = tl.reshape(qk, (TOKEN * GROUP, BLOCK_N), can_reorder=False)

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



@triton.jit
def compatible_single_seg_mask_attn_kernel(
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
        Mask,
        MASK_SIZE: tl.constexpr,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        TOKEN: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    seqlen_q = tl.load(q_lengths + bid)
    if mid * TOKEN >= seqlen_q:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)
    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid)
    seqlen_k = tl.load(k_lengths + bid)
    gap = seqlen_k - seqlen_q

    q_idx = offs_m // GROUP
    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q * H, (mid + 1) * TOKEN * H)

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

    mask_free_step = gap // BLOCK_N
    for i in range(0, mask_free_step):
        n = i * BLOCK_N

        k = tl.load(k_ptrs + n * stride_k)
        qk = tl.dot(q, tl.trans(k))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_k)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)


    sub_gap = gap - mask_free_step * BLOCK_N
    mask_ptrs = Mask + bid * MASK_SIZE * MASK_SIZE + \
             mid * TOKEN * MASK_SIZE - sub_gap + \
                 (tl.arange(0,BLOCK_M)%(GROUP*TOKEN)//GROUP)[:,None] * MASK_SIZE + offs_n[None,:]
    q_pos = (mid * TOKEN + gap) + q_idx[:, None]
    for n in range(mask_free_step * BLOCK_N, mid * TOKEN + TOKEN + gap, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        if EVEN_N:
            k = tl.load(k_ptrs + n * stride_k)
        else:
            k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
        qk = tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        qk += tl.where(q_pos >= (n + offs_n)[None, :], 0, float("-inf"))

        # mask
        mask = tl.load(mask_ptrs, mask=(offs_n[None, :] < MASK_SIZE) & (
                    offs_n[None, :] >= sub_gap), other=0.0)
        qk = tl.where(mask==0, qk, qk-10000.0)

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
    max_m_off = min(seqlen_q * H, (mid + 1) * TOKEN * H)

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
used for decode
"""


@triton.jit
def multi_seg_full_attn_kernel(
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
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
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

    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H
    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
                other=0.0)

    n_seg = tl.load(k_segs + bid)

    for i_seg in range(n_seg):

        k_offset = tl.load(k_offsets + bid * (max_seg + 0) + i_seg)
        seqlen_k = tl.load(
            k_lengths + bid * (max_seg + 1) + i_seg + 1) - tl.load(
            k_lengths + bid * (max_seg + 1) + i_seg)

        k_ptrs = (
                K + k_offset * stride_k + hid * HEADDIM + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )
        v_ptrs = (
                V + k_offset * stride_k + hid * HEADDIM + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )

        for n in range(0, seqlen_k, BLOCK_N):
            n = tl.multiple_of(n, BLOCK_N)

            k = tl.load(k_ptrs + n * stride_k)
            # k = tl.load(k_ptrs + n * stride_k, mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
            qk = tl.dot(q, tl.trans(k))
            if not EVEN_N:
                qk += tl.where((n + offs_n)[None, :] < seqlen_k, 0,
                               float("-inf"))

            p = tl.exp(qk * softmax_scale)

            lse_i += tl.sum(p, 1)

            v = tl.load(v_ptrs + n * stride_k)
            # v = tl.load(v_ptrs + n * stride_k, mask=(n + offs_n)[:, None] < seqlen_k, other=0.0)
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


"""
limitation: q length should be shorter than the last kv segment length
TODO: ADAPT FOR THE LIMITATION
"""


@triton.jit
def multi_seg_causal_attn_kernel(
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
        k_lengths,  # [0,l,2l,...,0,l,2l]
        k_segs,
        max_seg,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    seqlen_q = tl.load(q_lengths + bid)
    TOKEN = BLOCK_M // GROUP
    if mid * TOKEN >= seqlen_q:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)

    q_offset = tl.load(q_offsets + bid)
    n_seg = tl.load(k_segs + bid)
    seqlen_total_k = tl.load(k_lengths + bid * (max_seg + 1) + n_seg)

    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H
    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
                other=0.0)

    gap = seqlen_total_k - seqlen_q

    for i_seg in range(n_seg - 1):

        seqlen_k_accum = tl.load(k_lengths + bid * (max_seg + 1) + i_seg)
        seqlen_k_accum_next = tl.load(
            k_lengths + bid * (max_seg + 1) + i_seg + 1)
        k_offset = tl.load(k_offsets + bid * max_seg + i_seg)
        seqlen_k_seg = seqlen_k_accum_next - seqlen_k_accum

        k_ptrs = (
                (K + k_offset * stride_k + hid * HEADDIM) + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )
        v_ptrs = (
                (V + k_offset * stride_k + hid * HEADDIM) + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )

        for i in range((seqlen_k_seg - 1) // BLOCK_N + 1):
            n = i * BLOCK_N
            n = tl.multiple_of(n, BLOCK_N)
            if EVEN_N:
                k = tl.load(k_ptrs + n * stride_k)
            else:
                k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < seqlen_k_seg,
                            other=0.0)
            qk = tl.dot(q, tl.trans(k))

            if not EVEN_N:
                qk += tl.where((n + offs_n)[None, :] < seqlen_k_seg, 0,
                               float("-inf"))

            p = tl.exp(qk * softmax_scale)

            lse_i += tl.sum(p, 1)

            if EVEN_N:
                v = tl.load(v_ptrs + n * stride_k)
            else:
                v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < seqlen_k_seg,
                            other=0.0)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

    seqlen_k_accum = tl.load(k_lengths + bid * (max_seg + 1) + n_seg - 1)
    seqlen_k_accum_next = tl.load(k_lengths + bid * (max_seg + 1) + n_seg)
    k_offset = tl.load(k_offsets + bid * max_seg + n_seg - 1)
    seqlen_k_seg = seqlen_k_accum_next - seqlen_k_accum

    k_ptrs = (
            (K + k_offset * stride_k + hid * HEADDIM) + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
            (V + k_offset * stride_k + hid * HEADDIM) + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )

    mask_free_step = min(max(mid * TOKEN + gap - seqlen_k_accum, 0),
                         seqlen_k_seg) // BLOCK_N
    for i in range(mask_free_step):
        n = i * BLOCK_N
        n = tl.multiple_of(n, BLOCK_N)
        k = tl.load(k_ptrs + n * stride_k)
        qk = tl.dot(q, tl.trans(k))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_k)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

    q_pos = ((mid * TOKEN + gap) + tl.arange(0, BLOCK_M) // GROUP)[:, None]
    max_k_idx = min(mid * TOKEN + TOKEN + gap - seqlen_k_accum, seqlen_k_seg)
    for n in range(mask_free_step * BLOCK_N, max_k_idx, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        if EVEN_N:
            k = tl.load(k_ptrs + n * stride_k)
        else:
            k = tl.load(k_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k_seg, other=0.0)
        qk = tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((n + offs_n)[None, :] < seqlen_k_seg, 0,
                           float("-inf"))

        qk += tl.where(q_pos >= (seqlen_k_accum + n + offs_n)[None, :], 0,
                       float("-inf"))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        if EVEN_N:
            v = tl.load(v_ptrs + n * stride_k)
        else:
            v = tl.load(v_ptrs + n * stride_k,
                        mask=(n + offs_n)[:, None] < seqlen_k_seg, other=0.0)
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


@triton.jit
def multi_seg_mask_attn_kernel(
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
        k_lengths,  # [0,l,2l,...,0,l,2l]
        k_segs,
        max_seg,
        Mask,
        MASK_SIZE: tl.constexpr,
        HEADDIM: tl.constexpr,
        GROUP: tl.constexpr,
        TOKEN: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    seqlen_q = tl.load(q_lengths + bid)
    if mid * TOKEN >= seqlen_q:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEADDIM)

    q_offset = tl.load(q_offsets + bid)
    n_seg = tl.load(k_segs + bid)
    seqlen_total_k = tl.load(k_lengths + bid * (max_seg + 1) + n_seg)

    H = stride_q // HEADDIM
    offs_m = offs_m % GROUP + offs_m // GROUP * H
    max_m_off = min(seqlen_q, (mid + 1) * TOKEN) * H
    q_ptrs = (
            Q + q_offset * stride_q + mid * TOKEN * stride_q + hid * HEADDIM * GROUP + (
                offs_m[:, None] * HEADDIM + offs_d[None, :])
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEADDIM], dtype=tl.float32)

    gap = seqlen_total_k - seqlen_q

    q = tl.load(q_ptrs, mask=(mid * TOKEN * H + offs_m)[:, None] < max_m_off,
                other=0.0)

    for i_seg in range(n_seg - 1):

        seqlen_k_accum = tl.load(k_lengths + bid * (max_seg + 1) + i_seg)
        seqlen_k_accum_next = tl.load(
            k_lengths + bid * (max_seg + 1) + i_seg + 1)
        k_offset = tl.load(k_offsets + bid * max_seg + i_seg)
        seqlen_k_seg = seqlen_k_accum_next - seqlen_k_accum

        k_ptrs = (
                (K + k_offset * stride_k + hid * HEADDIM) + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )
        v_ptrs = (
                (V + k_offset * stride_k + hid * HEADDIM) + (
                    offs_n[:, None] * stride_k + offs_d[None, :])
        )

        for i in range((seqlen_k_seg - 1) // BLOCK_N + 1):
            n = i * BLOCK_N
            n = tl.multiple_of(n, BLOCK_N)
            if EVEN_N:
                k = tl.load(k_ptrs + n * stride_k)
            else:
                k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < seqlen_k_seg,
                            other=0.0)
            qk = tl.dot(q, tl.trans(k))

            if not EVEN_N:
                qk += tl.where((n + offs_n)[None, :] < seqlen_k_seg, 0,
                               float("-inf"))

            p = tl.exp(qk * softmax_scale)

            lse_i += tl.sum(p, 1)

            if EVEN_N:
                v = tl.load(v_ptrs + n * stride_k)
            else:
                v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_n)[:, None] < seqlen_k_seg,
                            other=0.0)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

    seqlen_k_accum = tl.load(k_lengths + bid * (max_seg + 1) + n_seg - 1)
    k_offset = tl.load(k_offsets + bid * max_seg + n_seg - 1)
    seqlen_k_seg = tl.load(
        k_lengths + bid * (max_seg + 1) + n_seg) - seqlen_k_accum

    k_ptrs = (
            (K + k_offset * stride_k + hid * HEADDIM) + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
            (V + k_offset * stride_k + hid * HEADDIM) + (
                offs_n[:, None] * stride_k + offs_d[None, :])
    )

    mask_free_step = (gap - seqlen_k_accum) // BLOCK_N
    for i in range(mask_free_step):
        n = i * BLOCK_N
        n = tl.multiple_of(n, BLOCK_N)
        k = tl.load(k_ptrs + n * stride_k)
        qk = tl.dot(q, tl.trans(k))

        p = tl.exp(qk * softmax_scale)

        lse_i += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_k)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

    q_pos = ((mid * TOKEN + gap) + tl.arange(0, BLOCK_M) // GROUP)[:, None]
    max_k_idx = min(mid * TOKEN + TOKEN + gap - seqlen_k_accum, seqlen_k_seg)
    sub_gap = gap - seqlen_k_accum - mask_free_step * BLOCK_N
    mask_ptrs = Mask + bid * MASK_SIZE * MASK_SIZE + mid * TOKEN * MASK_SIZE - sub_gap + (
                                                                                             tl.arange(
                                                                                                 0,
                                                                                                 TOKEN))[
                                                                                         :,
                                                                                         None] * MASK_SIZE + offs_n[
                                                                                                             None,
                                                                                                             :]
    for n in range(mask_free_step * BLOCK_N, max_k_idx, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        k = tl.load(k_ptrs + n * stride_k,
                    mask=(n + offs_n)[:, None] < seqlen_k_seg, other=0.0)
        qk = tl.dot(q, tl.trans(k))

        qk += tl.where((n + offs_n)[None, :] < seqlen_k_seg, 0, float("-inf"))

        qk += tl.where(q_pos >= (seqlen_k_accum + n + offs_n)[None, :], 0,
                       float("-inf"))

        # mask
        qk = tl.reshape(qk, (TOKEN, GROUP, BLOCK_N), can_reorder=False)
        mask = tl.load(mask_ptrs, mask=(offs_n[None, :] < MASK_SIZE) & (
                    offs_n[None, :] >= sub_gap), other=0.0)
        mask = -10000 * tl.cast(mask, tl.float32)
        qk += mask[:, None, :]
        qk = tl.reshape(qk, (TOKEN * GROUP, BLOCK_N), can_reorder=False)

        p = tl.exp(qk * softmax_scale)
        lse_i += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_k,
                    mask=(n + offs_n)[:, None] < seqlen_k_seg, other=0.0)
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


def seg_attn_fwd(q, k, v, meta, causal=False):
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
        [x % BLOCK_M == 0 and BLOCK_M % GROUP == 0 for
         x in meta.qls])
    if isinstance(meta.kls[0], (list, tuple)):
        EVEN_N = all(
            [all([y % BLOCK_N == 0 for y in x]) for x in meta.kls])
    else:
        EVEN_N = all([x % BLOCK_N == 0 for x in meta.kls])

    # _CudaDeviceProperties(name='NVIDIA H20', major=9, minor=0, total_memory=97285MB, multi_processor_count=78)
    dp = torch.cuda.get_device_properties(0)
    sm = dp.major * 10 + dp.minor

    TOKEN = BLOCK_M // GROUP
    num_m_block = (meta.max_q_length - 1) // TOKEN + 1
    num_warps = 8  # TODO: only for compatible mode
    if meta.mask is not None and BLOCK_M % GROUP != 0:
        num_stages = 1
    elif meta.mask is not None and BLOCK_M % GROUP == 0:
        num_stages = 2
    else:
        num_stages = 3
    if 80 < sm <= 89:
        num_stages = max(1, num_stages - 1)
    grid = lambda META: (batch, kv_heads, num_m_block)
    if SINGLE:
        if CAUSAL:
            if meta.mask is None:
                single_seg_causal_attn_kernel[grid](
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

            else:
                kernel = single_seg_mask_attn_kernel if BLOCK_M % GROUP == 0 else compatible_single_seg_mask_attn_kernel
                kernel[grid](
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
                    meta.mask,
                    meta.mask.size(-1),
                    HEADDIM,
                    GROUP,
                    TOKEN,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    EVEN_M=EVEN_M,
                    EVEN_N=EVEN_N,
                    num_warps=num_warps,
                    num_stages=num_stages
                )
                return o
        else:
            single_seg_full_attn_kernel[grid](
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

    else:
        if CAUSAL:
            if meta.mask is None:

                multi_seg_causal_attn_kernel[grid](
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
                    HEADDIM,
                    GROUP,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    EVEN_M=EVEN_M,
                    EVEN_N=EVEN_N,
                    num_warps=num_warps,
                    num_stages=num_stages
                )
                return o
            else:
                multi_seg_mask_attn_kernel[grid](
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
                    meta.mask,
                    meta.mask.size(-1),
                    HEADDIM,
                    GROUP,
                    TOKEN,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    EVEN_M=EVEN_M,
                    EVEN_N=EVEN_N,
                    num_warps=num_warps,
                    num_stages=num_stages
                )
                return o
        else:
            multi_seg_full_attn_kernel[grid](
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
                HEADDIM,
                GROUP,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return o
