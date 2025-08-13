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
        decay_scales,
        # Mask,
        HEAD_DIM: tl.constexpr,
        SPLIT_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
        EVEN: tl.constexpr,
        DECOUPLE: tl.constexpr,
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
    decay_scale = tl.load(decay_scales + hid)
    s_scale = tl.load(s_scales+bid)

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

    state = tl.load(s_ptrs).to(tl.float32)
    state = state*s_scale.to(state.dtype)  # s_scale is 0 or 1, cast is precision preserved

    if BLOCK > 1:
        for n in range(0, q_length, BLOCK):
            n = tl.multiple_of(n, BLOCK)

            if EVEN:
                q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
                k = tl.load(k_ptrs + n * stride_k).to(tl.float32)
                v = tl.load(v_ptrs + n * stride_k).to(tl.float32)
            else:
                q = tl.load(q_ptrs + n * stride_q,
                            mask=(n + offs_m)[:, None] < q_length, 
                            other=0.0).to(tl.float32)
                k = tl.load(k_ptrs + n * stride_k,
                            mask=(n + offs_m)[:, None] < q_length, 
                            other=0.0).to(tl.float32)
                v = tl.load(v_ptrs + n * stride_k,
                            mask=(n + offs_m)[:, None] < q_length, 
                            other=0.0).to(tl.float32)

            if DECOUPLE:
                # only work with small scales
                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                b_offs = b-1-offs_m

                decays = tl.where(b_offs >= 0, tl.exp(decay_scale*b_offs), 0)
                inv_decays = tl.where(b_offs >= 0, tl.exp(-decay_scale*b_offs), 0)

                q = q*inv_decays[:,None]
                k = k*decays[:,None]
                qk = tl.dot(q, tl.trans(k)) * softmax_scale
                qk = tl.where(offs_m[None,:] <= offs_m[:,None], qk, 0.0)
                o = tl.dot(qk, v)

                block_decay_plus = tl.exp(decay_scale*b) * softmax_scale
                o = tl.dot(q, state)*block_decay_plus + o

                block_decay = tl.exp(decay_scale*b)
                state = state * block_decay + tl.dot(tl.trans(k), v)
            else:

                qk = tl.dot(q, tl.trans(k)) * softmax_scale
                decays = tl.exp(decay_scale*(offs_m[:,None] - offs_m[None,:]))
                decays = tl.where(offs_m[None,:] <= offs_m[:,None], decays, 0.0)
                qk *= decays
                o = tl.dot(qk, v) 

                decay_arr = tl.exp(decay_scale*(offs_m[:,None]+1)) * softmax_scale
                o = tl.dot(q*decay_arr, state, acc=o)

                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                b_offs = b-1-offs_m
                b_offs = tl.where(b_offs >= 0, b_offs, 10000)
                decays = tl.exp(decay_scale*b_offs)
                block_decay = tl.exp(decay_scale*b)
                state = state * block_decay + tl.dot(tl.trans(k*decays[:,None]), v.to(tl.float32))

            if EVEN:
                tl.store(out_ptrs + n * stride_o, o.to(Out.dtype.element_ty))
            else:
                tl.store(out_ptrs + n * stride_o, o.to(Out.dtype.element_ty), mask=(n + offs_m)[:, None] < q_length)

        tl.store(s_ptrs, state.to(S.dtype.element_ty))

    else:
        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        state = state * tl.exp(decay_scale) + tl.trans(k) * v

        o = tl.sum(tl.trans(q) * state, axis=0, keep_dims=True) * softmax_scale.to(q.dtype)

        tl.store(out_ptrs, o.to(Out.dtype.element_ty))

        tl.store(s_ptrs, state.to(S.dtype.element_ty))

def seg_la_fwd(q, k, v, s, decay_scales, meta, softmax_scale=None, decouple=False):
    _, qo_heads, d = q.shape
    _, kv_heads, _ = k.shape
    batch = meta.batch_size
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    HEAD_DIM = d

    GROUP = qo_heads // kv_heads

    # NOT support GQA currently
    assert GROUP == 1
    # NOT support customized MASK currently
    assert meta.mask is None

    DECOUPLE = decouple

    BLOCK = 1 if meta.max_q_length == 1 else (64 if DECOUPLE else HEAD_DIM)

    EVEN = all([x % BLOCK == 0 for x in meta.qls])

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

    SPLIT_DIM = 64 if BLOCK == 1 else 32
    num_dim_block = HEAD_DIM // SPLIT_DIM
    num_warps = 8
    num_stages = 3 if DECOUPLE else 2

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
        decay_scales,
        # meta.mask,
        HEAD_DIM=HEAD_DIM,
        SPLIT_DIM=SPLIT_DIM,
        BLOCK=BLOCK,
        EVEN=EVEN,
        DECOUPLE=DECOUPLE,
        # MASK_SIZE=MASK_SIZE,
        # MASK_TYPE=MASK_TYPE,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o