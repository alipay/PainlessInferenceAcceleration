# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import time
import random
import numpy as np
import math

import torch
try:
    import flash_attn_2_cuda
except:
    flash_attn_2_cuda = None
try:
    import flash_attn_3_cuda
except:
    flash_attn_3_cuda = None




def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def benchmark_func(fn, *args, n_warmup=100, n_repeat=1000, ref_time=None,
                   ref_flops=None, desc='', **kwargs):
    func_name = fn.__name__

    for i in range(n_warmup):
        fn(*args, **kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in
                    range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in 
                    range(n_repeat)]

    for i in range(n_repeat):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1, n_repeat // 100)
    times = sum(times[clip:-clip])

    average_event_time = times * 1000 / (n_repeat - 2 * clip)

    line = f'{func_name:<32} {desc} time:{average_event_time:.1f} us'

    if ref_flops is not None:
        perf = ref_flops / 1e12 / (
                average_event_time / 1e6)
        line += f' perf:{perf:.2f} TFLOPS'

    if ref_time is not None:
        speedup = ref_time / average_event_time 
        line += f' speedup:{speedup:.2f}'
    
    print(line)
    return average_event_time


def torch_attn(q, k, v, causal=True, mask=None):
    bs, q_len, q_head, head_dim = q.shape
    k_head = k.shape[2]
    k_len = k.shape[1]
    if mask is None:
        if causal:
            mask = -10000 * torch.triu(
                torch.ones((q_len, k_len), dtype=q.dtype, device='cuda:0'),
                k_len - q_len + 1)
        else:
            mask = torch.zeros((q_len, k_len), dtype=q.dtype, device='cuda:0')

    query = q.transpose(1, 2)
    key = torch.permute(k, (0, 2, 3, 1))
    value = v.transpose(1, 2)
    if k_head != q_head:
        g = q_head // k_head
        key = torch.repeat_interleave(key, g, dim=1)
        value = torch.repeat_interleave(value, g, dim=1)
    score = torch.matmul(query, key) / math.sqrt(head_dim) + mask
    prob = torch.softmax(score, dim=-1, dtype=torch.float32).to(q.dtype)
    att = torch.matmul(prob, value)
    att = torch.reshape(att.transpose(1, 2),
                        [bs, q_len, q_head, head_dim]).contiguous()
    return att


def flash_attn_2(q, k, v, meta):
    softmax_scale = q.size(-1) ** (-0.5)
    causal = True
    outputs = flash_attn_2_cuda.varlen_fwd(
        q,
        k,
        v,
        None,  # out_
        meta.cu_seqlens_q,
        meta.cu_seqlens_k,
        meta.seqused_k,
        None,  # leftpad_k
        None,  # block_table
        None,  # alibi_slopes
        meta.max_seqlen_q,
        meta.max_seqlen_k,
        0.0,  # dropout
        softmax_scale,
        False,  # zero_tensors
        causal,  # causal 
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        False,  # return_softmax
        None  # Generator
    )

    return outputs[0]


def flash_attn_3(q, k, v, meta):
    softmax_scale = q.size(-1) ** (-0.5)
    causal = True
    # outputs = flashattn_hopper_cuda.varlen_fwd(
    #     q,
    #     k,
    #     v,
    #     None,  # out_
    #     meta.cu_seqlens_q,
    #     meta.cu_seqlens_k,
    #     None,  # meta.seqused_q,
    #     meta.seqused_k,
    #     meta.max_seqlen_q,
    #     meta.max_seqlen_k,
    #     softmax_scale,
    #     True,  # causal 
    #     -1,  # window_size_left
    #     -1,  # window_size_right
    # )
    # return outputs[0]
    pack_gqa = True
    out, softmax_lse, *rest = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        meta.cu_seqlens_q,
        meta.cu_seqlens_k,
        None,
        meta.seqused_q,
        meta.seqused_k,
        meta.max_seqlen_q,
        meta.max_seqlen_k,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        False,
        0,
        pack_gqa,
        0
    )
    return out



class Meta:
    def __init__(self) -> None:
        pass


def split(kls, max_seg):
    outputs = []
    for kl in kls:
        if max_seg == 1:
            outputs.append([kl])
        else:
            segs = [kl // max_seg] * (max_seg - 1)
            segs.append(kl - sum(segs))
            outputs.append(segs)
    return outputs


def get_seg_attn_meta(qls, klss, mask=None):
    device = 'cuda:0'
    bs = len(qls)
    max_seg = max([len(x) for x in klss])  # equals
    q_offsets = [0]  # [bs+1]
    k_offsets = [0]  # [bs+1] if single seg else [bs, max_seg]
    q_lengths = []  # [bs]
    k_lengths = []  # [bs] if single seg else [bs, max_seg+1]
    k_segs = []
    max_q_length = max(qls)
    max_k_length = max([sum(x) for x in klss])
    for i, ql in enumerate(qls):
        kls = klss[i]
        q_offsets.append(q_offsets[-1] + ql)
        q_lengths.append(ql)
        for j, kl in enumerate(kls):
            k_offsets.append(k_offsets[-1] + kl)
            k_lengths.append(kl)
        if max_seg > 1:
            k_lengths.append(sum(kls))
        k_segs.append(max_seg)

    q_offsets = torch.tensor(q_offsets, device=device, dtype=torch.int32)
    k_offsets = torch.tensor(k_offsets, device=device, dtype=torch.int32)
    q_lengths = torch.tensor(q_lengths, device=device, dtype=torch.int32)
    k_lengths = torch.tensor(k_lengths, device=device, dtype=torch.int32)
    k_segs = torch.tensor(k_segs, device=device, dtype=torch.int32)

    meta = Meta()
    meta.batch_size = bs
    meta.q_offsets = q_offsets
    meta.k_offsets = k_offsets
    meta.q_lengths = q_lengths
    meta.k_lengths = k_lengths
    meta.k_segs = k_segs
    meta.max_q_length = max_q_length
    meta.max_k_length = max_k_length
    meta.max_seg = max_seg
    meta.mask = mask
    meta.qls = qls
    meta.kls = klss
    return meta


def get_flash_attn_meta(qls, kls, mask=None):
    device = 'cuda:0'

    cu_seqlens_q = [0]  # bs+1
    cu_seqlens_k = [0]  # bs+1
    seqused_q = []  # bs
    seqused_k = []  # bs
    max_seqlen_q = max(qls)
    max_seqlen_k = max(kls)
    for i, ql in enumerate(qls):
        kl = kls[i]
        cu_seqlens_q.append(cu_seqlens_q[-1] + ql)
        cu_seqlens_k.append(cu_seqlens_k[-1] + kl)
        seqused_q.append(ql)
        seqused_k.append(kl)

    cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, device=device, dtype=torch.int32)
    seqused_q = torch.tensor(seqused_q, device=device, dtype=torch.int32)
    seqused_k = torch.tensor(seqused_k, device=device, dtype=torch.int32)

    meta = Meta()
    meta.batch_size = len(qls)
    meta.cu_seqlens_q = cu_seqlens_q
    meta.cu_seqlens_k = cu_seqlens_k
    meta.seqused_q = seqused_q
    meta.seqused_k = seqused_k
    meta.max_seqlen_q = max_seqlen_q
    meta.max_seqlen_k = max_seqlen_k
    meta.mask = mask
    return meta
