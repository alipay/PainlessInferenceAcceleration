# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl


"""
MASK_TYPE
     0: full mask
     1: causal mask
     2: customized mask
"""

@triton.jit
def seg_la_kernel(
        Q,
        K,
        V,
        S,
        Out,
        softmax_scale,
        stride_q,
        stride_k,
        stride_o,
        stride_s,
        s_offsets,
        q_offsets,
        q_lengths,
        s_scales,
        # Mask,
        HEAD_DIM: tl.constexpr,
        SPLIT_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
        EVEN: tl.constexpr,
        # MASK_SIZE: tl.constexpr,
        # MASK_TYPE: tl.constexpr
):  
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLIT_DIM)

    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)

    q_ptrs = (
            Q + q_offset * stride_q + hid * HEAD_DIM + (
                offs_m[:, None] * stride_q + offs_d[None, :])
    )
    k_ptrs = (
            K + q_offset * stride_k + hid * HEAD_DIM + (
                offs_m[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
            V + q_offset * stride_k + hid * HEAD_DIM + sid * SPLIT_DIM +  (
                offs_m[:, None] * stride_k + offs_s[None, :])
    )
    out_ptrs = (
            Out + q_offset * stride_o + hid * HEAD_DIM + sid * SPLIT_DIM + (
                offs_m[:, None] * stride_o + offs_s[None, :])
    )
    s_ptrs =  (
            S + s_offset * stride_s + hid * HEAD_DIM * HEAD_DIM + sid * SPLIT_DIM + (
                offs_d[:, None] * HEAD_DIM + offs_s[None, :])
    )


    state = tl.load(s_ptrs)
    s_scale = tl.load(s_scales+bid)
    state = (state*s_scale).to(S.dtype.element_ty)

    # if bid == 0:
    #     if hid == 1:
    #         if sid == 1:
    #             tl.device_print("state",state.to(tl.float32))

    if BLOCK > 1:
        for n in range(0, q_length, BLOCK):
            n = tl.multiple_of(n, BLOCK)

            if EVEN:
                q = tl.load(q_ptrs + n * stride_q)
                k = tl.load(k_ptrs + n * stride_k)
                v = tl.load(v_ptrs + n * stride_k)
            else:
                q = tl.load(q_ptrs + n * stride_q,
                            mask=(n + offs_m)[:, None] < q_length, 
                            other=0.0)
                k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_m)[:, None] < q_length, 
                            other=0.0)
                v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_m)[:, None] < q_length, 
                            other=0.0)

            # k = (k * softmax_scale).to(q.dtype)
            qk = tl.dot(q, tl.trans(k)) * softmax_scale
            qk = tl.where(offs_m[:,None]<=offs_m[None,:],qk,0.0)
            o = tl.dot(qk.to(v.dtype), v) + tl.dot(q, state)
            state += (tl.dot(tl.trans(k), v) * softmax_scale).to(S.dtype.element_ty)

            if EVEN:
                tl.store(out_ptrs + n * stride_o, o)
            else:
                tl.store(out_ptrs + n * stride_o, o, mask=(n + offs_m)[:, None] < q_length)

    else:
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        state += ((tl.trans(k) * v)*softmax_scale).to(S.dtype.element_ty)
        o = tl.sum(tl.trans(q) * state, axis=0, keep_dims=True)

        tl.store(out_ptrs, o)

    tl.store(s_ptrs, state)



def seg_la_fwd(q, k, v, s, meta):
    _, qo_heads, d = q.shape
    _, kv_heads, _ = k.shape
    batch = meta.batch_size
    softmax_scale = 1.0 / math.sqrt(d)

    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    HEAD_DIM = d

    GROUP = qo_heads // kv_heads

    # NOT support GQA currently
    assert GROUP == 1
    # NOT support customized MASK currently
    assert meta.mask is None

    BLOCK = 1 if meta.max_q_length == 1 else HEAD_DIM

    EVEN = all(
        [x % BLOCK == 0 for
         x in meta.qls])

    # if meta.mask is not None:
    #     MASK_TYPE = 2
    #     MASK_SIZE = meta.mask.size(-1)
    #     assert all([x==MASK_SIZE for x in meta.qls])
    # elif meta.max_q_length > 1:
    #     MASK_TYPE = 1
    #     MASK_SIZE = 0
    # else:
    #     MASK_TYPE = 0
    #     MASK_SIZE = 0

    SPLIT_DIM = 16
    num_dim_block = HEAD_DIM // SPLIT_DIM
    num_warps = 8
    if meta.mask is None:
        num_stages = 3
    else:
        num_stages = 2

    # name='NVIDIA H20', major=9, minor=0, total_memory=97285MB, multi_processor_count=78
    prop = torch.cuda.get_device_properties(0)
    sm = prop.major * 10 + prop.minor
    if sm not in (80, 90):
        num_stages = max(1, num_stages - 1)

    grid = lambda META: (batch, kv_heads, num_dim_block)
    seg_la_kernel[grid](
        q,
        k,
        v,
        s,
        o,
        softmax_scale,
        q.stride(0),
        k.stride(0),
        o.stride(0),
        s.stride(0),
        meta.s_offsets,
        meta.q_offsets,
        meta.q_lengths,
        meta.s_scales,
        # meta.mask,
        HEAD_DIM=HEAD_DIM,
        SPLIT_DIM=SPLIT_DIM,
        BLOCK=BLOCK,
        EVEN=EVEN,
        # MASK_SIZE=MASK_SIZE,
        # MASK_TYPE=MASK_TYPE,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o