# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl

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



def scaled_dot_product_attention(query, key_value):
    # query:[bs, q_length, 128, 576]
    # key_vaue: [bs, k_length, 576]
    _, q_length, _, q_dim = query.size()
    k_length = key_value.size(1)
    query = query.float()
    key = key_value.float()
    value = key_value[:,:,:512].float()
    query = query.permute(0,2,1,3)  # [bs, 128, q_length, 576]
    key = key.unsqueeze(1).permute(0,1,3,2)  # [bs, 1, 576, k_length]
    value = value.unsqueeze(1)   # [bs, 1, k_length, 512]
    attn_weight = query @ key / math.sqrt(q_dim)  # [bs, 128, q_length, k_length]
    mask = torch.tril(torch.ones(q_length, k_length, dtype=torch.float32, device=query.device), k_length-q_length)
    # print(mask)
    attn_weight -= 10000*(1-mask)
    lse = torch.exp(attn_weight).sum(-1).permute(0,2,1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    output = attn_weight @ value  # [bs, 128, q_length, 512]
    output = output.permute(0,2,1,3).contiguous()
    return output, lse


# def scaled_dot_product_attention(query, key_value):
#     # query:[bs, q_length, 128, 576]
#     # key_vaue: [bs, k_length, 576]
#     _, q_length, _, q_dim = query.size()
#     k_length = key_value.size(1)
#     query = query.clone()
#     key = key_value.clone()
#     value = key_value[:,:,:512].clone()
#     query = query.permute(0,2,1,3)  # [bs, 128, q_length, 576]
#     key = key.unsqueeze(1).permute(0,1,3,2)  # [bs, 1, 576, k_length]
#     value = value.unsqueeze(1)   # [bs, 1, k_length, 512]
#     attn_weight = query @ key / math.sqrt(q_dim)  # [bs, 128, q_length, k_length]
#     mask = torch.tril(torch.ones(q_length, k_length, dtype=query.dtype, device=query.device), k_length-q_length)
#     # print(mask)
#     attn_weight -= 10000*(1-mask)
#     lse = torch.exp(attn_weight).sum(-1).permute(0,2,1)
#     attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query.dtype)
#     output = attn_weight @ value  # [bs, 128, q_length, 512]
#     output = output.permute(0,2,1,3).contiguous()
#     return output, lse


@triton.jit
def batch_mla_kernel(
        Q,
        KV,
        Out,
        LSE,
        softmax_scale,
        q_length,
        k_length,
        BLOCK: tl.constexpr,
        EVEN: tl.constexpr,
):
    # q: [bs, q_len, 128, 576]
    # kv: [bs, k_len, 576]

    bid = tl.program_id(0)
    mid = tl.program_id(1)

    offs_v = tl.arange(0, 512)
    offs_p = tl.arange(0, 64)
    offs_h = tl.arange(0, 128)
    offs_n = tl.arange(0, BLOCK)

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
    # if bid == 0:
    #     if mid == 1:
    #         tl.device_print("steps",steps)

    for step in range(steps):
        n = step * BLOCK

        if EVEN:
            k0 = tl.load(k0_ptrs + n * 576)
        else:
            k0 = tl.load(k0_ptrs + n * 576,
                         mask=(n + offs_n)[:, None] < max_n, other=0.0)
        qk = tl.dot(q0, tl.trans(k0))

        if EVEN:
            k1 = tl.load(k1_ptrs + n * 576 + 512)
        else:
            k1 = tl.load(k1_ptrs + n * 576 + 512,
                        mask=(n + offs_n)[:, None] < max_n, other=0.0)

        qk += tl.dot(q1, tl.trans(k1))

        # pad mask
        qk += tl.where((n + offs_n)[None, :] < max_n, 0.0, float("-inf"))

        p = tl.exp(qk * softmax_scale)

        lse += tl.sum(p, 1)

        if bid == 0:
            if mid == 0:
                if step == 15:
                    tl.device_print("p",p)

        p = p.to(k0.dtype)
        acc_o += tl.dot(p, k0)

    acc_o = (acc_o / lse[:, None]).to(Out.dtype.element_ty)

    # offs_h = tl.arange(0, 128)
    # offs_v = tl.arange(0, 512)
    out_ptrs = (
            Out + bid * q_length * 128 * 512 + mid * 128 * 512 + (
                offs_h[:, None] * 512 + offs_v[None, :])
    )

    tl.store(out_ptrs, acc_o)

    lse_ptrs = LSE + bid * q_length * 128 + mid * 128 + offs_h
    tl.store(lse_ptrs, lse)


def batch_mla_fwd(q, kv, causal=True):
    # q: [bs, q_len, 128, 576]
    # kv: [bs, k_len, 576]
    assert causal
    bs, q_length, q_heads, q_d = q.shape
    bs, k_length, k_d = kv.shape
    assert q_d == k_d
    softmax_scale = 1.0 / math.sqrt(q_d)

    v_d = 512
    o = torch.empty((bs, q_length, q_heads, v_d), device=q.device, dtype=q.dtype)
    lse = torch.empty((bs, q_length, q_heads), device=q.device, dtype=torch.float32)

    BLOCK = 32

    EVEN = k_length % BLOCK == 0

    num_m_block = q_length
    num_warps = 8
    num_stages = 1
    grid = lambda META: (bs, num_m_block)
    batch_mla_kernel[grid](
        q,
        kv,
        o,
        lse,
        softmax_scale,
        q_length,
        k_length,
        BLOCK=BLOCK,
        EVEN=EVEN,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o, lse


if __name__ == '__main__':
    device = 'cuda:0'
    dtype = torch.bfloat16 
    bs = 1 
    q_length = 2
    k_length = 1024
    q = torch.randn((bs,q_length,128,576), device=device, dtype=dtype)
    kv = torch.randn((bs,k_length,576), device=device, dtype=dtype)

    ref_output, ref_lse = scaled_dot_product_attention(q, kv)
    opt_output, opt_lse = batch_mla_fwd(q, kv)

    # print(f'{ref_output.shape=} {opt_output.shape=}')

    output_err = ((ref_output-opt_output.float()).abs().mean()/ref_output.abs().mean()).item()
    lse_err = ((ref_lse-opt_lse.float()).abs().mean()/ref_lse.abs().mean()).item()

    print(f"\noutput_err:{output_err:.3f} lse_err:{lse_err:.3f}\n")
    print(ref_output[0,0,0,:4])
    print(opt_output[0,0,0,:4])
    # print(ref_output[0,0,-1,:4])
    # print(opt_output[0,0,-1,:4])
    # print(ref_output[0,-1,0,:4])
    # print(opt_output[0,-1,0,:4])
    # benchmark_func(batch_mla_fwd, q, kv, n_repeat=100, ref_flops=bs*q_length*k_length*128*(576+512)*2/2)