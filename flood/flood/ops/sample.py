# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def sample_from_logit_kernel(
    topk_values,
    topk_indices,
    temperature,
    top_k,
    top_p,
    min_p,
    buffers,
    outputs,
    max_top_k,
    B: tl.constexpr,
):
    bid = tl.program_id(0)
    k = tl.load(top_k + bid)
    if k == 1:
        token_id = tl.load(topk_indices + bid * max_top_k)
        tl.store(outputs + bid, token_id)
        return

    p = tl.load(top_p + bid)
    t = tl.load(temperature + bid)
    mp = tl.load(min_p + bid)
    indices = tl.arange(0, B)
    values = tl.load(
        topk_values + bid * max_top_k + indices, mask=indices < k, other=-10000.0
    )
    values = values / t

    others = tl.full((1,), 0.0, dtype=values.dtype)

    max_values = tl.max(values)
    exps = tl.exp(values - max_values)
    sums = tl.sum(exps, axis=0)
    probs = exps / sums
    probs = tl.where(probs >= mp, probs, others)
    probs = probs / tl.sum(probs, axis=0)

    accum_probs = tl.cumsum(probs, reverse=True)

    probs = tl.where(accum_probs >= p, probs, others)
    probs = probs / tl.sum(probs)

    tl.store(buffers + bid * B + indices, probs)
    v = tl.rand(0, 0)

    acc = 0.0
    stop = False
    for i in range(0, k):
        if not stop:
            prob = tl.load(buffers + bid * B + i)
            if v >= acc:
                if v <= acc + prob:
                    token_id = tl.load(topk_indices + bid * max_top_k + i)
                    tl.store(outputs + bid, token_id)
                    stop = True
            acc += prob

    if not stop:
        token_id = tl.load(topk_indices + bid * max_top_k)
        tl.store(outputs + bid, token_id)


def sample_from_logit(logits, temperature, top_k, top_p, min_p, max_top_k):
    bs = logits.size(0)
    B = 2 ** int(math.log2(max_top_k - 1) + 1)
    topk_values, topk_indices = torch.topk(
        logits, max_top_k, dim=-1, largest=True, sorted=True
    )
    outputs = torch.zeros((bs,), dtype=topk_indices.dtype, device=logits.device)
    buffers = -10000 * torch.ones((bs, B), dtype=logits.dtype, device=logits.device)
    grid = lambda META: (bs,)
    sample_from_logit_kernel[grid](
        topk_values,
        topk_indices,
        temperature,
        top_k,
        top_p,
        min_p,
        buffers,
        outputs,
        max_top_k,
        B=B,
        num_warps=1,
        num_stages=1,
    )
    return outputs
