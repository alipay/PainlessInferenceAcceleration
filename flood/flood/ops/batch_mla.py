# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl

torch.cuda.random.manual_seed(7)
from flood.utils.benchmark import benchmark_func


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
def batch_mla_kernel(
        Q,
        KV,
        Out,
        softmax_scale,
        q_length,
        k_length,
        BLOCK: tl.constexpr,
        EVEN: tl.constexpr,
):
    # q: [bs, q_len, 128, 576]
    # kv: [bs, k_len, 576]

    bid = tl.program_id(0)  # batch size index
    mid = tl.num_programs(1) - tl.program_id(1) - 1  # query token index

    offs_n = tl.arange(0, BLOCK)
    offs_h = tl.arange(0, 128)
    offs_v = tl.arange(0, 512)
    offs_p = tl.arange(0, 64)

    q0_ptrs = (
            Q + bid * q_length * 128 * 576 + mid * 128 * 576 + (
                offs_h[:, None] * 576 + offs_v[None, :])
    )
    q1_ptrs = (
            Q + bid * q_length * 128 * 576 + mid * 128 * 576 + 512 + (
                offs_h[:, None] * 576 + offs_p[None, :])
    )
    k0_ptrs = (
            KV + bid * k_length * 576 + (
                offs_n[:, None] * 576 + offs_v[None, :])
    )
    k1_ptrs = (
            KV + bid * k_length * 576 + 512 + (
                offs_n[:, None] * 576 + offs_p[None, :])
    )

    lse = tl.zeros([128], dtype=tl.float32)
    acc_o = tl.zeros([128, 512], dtype=tl.float32)

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q1_ptrs)

    max_n = k_length - q_length + mid + 1

    steps = tl.cdiv(max_n, BLOCK)


    for step in range(steps):
        n = step * BLOCK

        if EVEN:
            k0 = tl.load(k0_ptrs + n * 576)
        else:
            k0 = tl.load(k0_ptrs + n * 576,
                         mask=(n + offs_n)[:, None] < max_n, other=0.0)
        qk = tl.dot(q0, tl.trans(k0))

        if EVEN:
            k1 = tl.load(k1_ptrs + n * 576)
        else:
            k1 = tl.load(k1_ptrs + n * 576,
                        mask=(n + offs_n)[:, None] < max_n, other=0.0)

        qk = tl.dot(q1, tl.trans(k1), qk)

        # pad mask
        qk += tl.where((n + offs_n)[None, :] < max_n, 0.0, float("-inf"))

        p = tl.exp(qk * softmax_scale)
        lse += tl.sum(p, 1)

        p = p.to(KV.dtype.element_ty)
        acc_o = tl.dot(p, k0, acc_o)

    acc_o = (acc_o / lse[:, None]).to(Out.dtype.element_ty)

    out_ptrs = (
            Out + bid * q_length * 128 * 512 + mid * 128 * 512 + (
                offs_h[:, None] * 512 + offs_v[None, :])
    )
    tl.store(out_ptrs, acc_o)

    # lse_ptrs = LSE + bid * q_length * 128 + mid * 128 + offs_h
    # tl.store(lse_ptrs, lse)


def batch_mla_fwd(q, kv):
    # q: [bs, q_len, 128, 576]
    # kv: [bs, k_len, 576]
    bs, q_length, q_heads, q_d = q.shape
    bs, k_length, k_d = kv.shape
    assert q_d == k_d
    softmax_scale = 1.0 / math.sqrt(q_d)

    v_d = 512
    o = torch.empty((bs, q_length, q_heads, v_d), device=q.device, dtype=q.dtype)
    lse = torch.empty((bs, q_length, q_heads), device=q.device, dtype=torch.float32)

    # print(f'{q.stride()=} {kv.stride()=} {o.stride()=}')

    BLOCK = 64

    EVEN = k_length % BLOCK == 0

    num_m_block = q_length
    num_warps = 8
    num_stages = 1
    grid = lambda META: (bs, num_m_block)
    batch_mla_kernel[grid](
        q,
        kv,
        o,
        softmax_scale,
        q_length,
        k_length,
        BLOCK=BLOCK,
        EVEN=EVEN,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o, lse

