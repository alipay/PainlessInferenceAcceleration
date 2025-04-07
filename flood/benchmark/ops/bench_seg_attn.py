# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

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

from flood.ops.segattn import seg_attn_fwd
from flood.utils.benchmark import benchmark_func

torch.manual_seed(7)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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


def flash_attn_2(q, k, v, meta, causal=True):
    softmax_scale = q.size(-1) ** (-0.5)

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


def flash_attn_3(q, k, v, meta, causal=True):
    softmax_scale = q.size(-1) ** (-0.5)


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


def seg_attn(q, k, v, meta, causal=True):
    outputs = seg_attn_fwd(
        q,
        k,
        v,
        meta,
        causal=causal
    )

    return outputs


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
    q_offsets = [0]  # bs+1
    k_offsets = [0]  # (bs*max_seg+1), [0,l,2l,] even if single seg
    q_lengths = []  # bs
    k_lengths = [
        0] if max_seg > 1 else []  # (bs*max_seg+bs) if multi seg, (bs+1) if single seg. multi seg format:[0,l,2l.,,,0,l,2l..,0]
    k_segs = []
    max_q_length = max(qls)
    max_k_length = max([sum(x) for x in klss])
    for i, ql in enumerate(qls):
        kls = klss[i]
        q_offsets.append(q_offsets[-1] + ql)
        q_lengths.append(ql)
        for j, kl in enumerate(kls):
            k_offsets.append(k_offsets[-1] + kl)
            k_lengths.append(k_lengths[-1] + kl if max_seg > 1 else kl)
        if max_seg > 1 and i < len(qls) - 1:
            k_lengths.append(0)
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


def bench_seg_attn(max_seg=1, mode='prefill', even=True, causal=True):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    qo_head = 64
    kv_head = 8
    group = qo_head // kv_head
    dim = 128
    mask_size = 16
    mask = False
    if mode == 'prefill':
        if max_seg == 1:
            if even:
                qls = [1024]
                kls = [1024]
            else:
                qls = [1025]
                kls = [1025]
        else:
            if even:
                qls = [1024]
                kls = [1024 * max_seg]
            else:
                qls = [1025]
                kls = [1025 * max_seg]
    elif mode == 'decode':
        qls = [1] * 128
        if even:
            kls = [1024] * 128
        else:
            kls = [1025] * 128
    elif mode == 'mix':
        if max_seg == 1:
            qls = [1024 - 39] + [1] * 39
        else:
            qls = [(1024 - 39) // max_seg] + [1] * 39
        if even:
            kls = [1024] * 40
        else:
            kls = [1025] * 40
    elif mode == 'spec':
        mask = True
        qls = [mask_size] * 1
        if even:
            kls = [1024] * 1
        else:
            kls = [1025] * 1
    else:
        raise ValueError(f'unknown mode:{mode}')

    klss = split(kls, max_seg)
    flops = 0

    flash_attn = flash_attn_3 if torch.cuda.get_device_properties(
            0).major >= 9 else flash_attn_2

    if mask:
        attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.uint8,
                                device=device)
        for i in range(mask_size):
            attn_mask[i, i + 1:] = 1
    else:
        attn_mask = None

    qs = []
    ks = []
    vs = []
    flops = 0.0
    for i, ql in enumerate(qls):
        kvl = kls[i]
        q = torch.randn(ql, qo_head, dim, dtype=dtype, device=device)
        k = torch.randn(kvl, kv_head, dim, dtype=dtype, device=device)
        v = torch.randn(kvl, kv_head, dim, dtype=dtype, device=device)
        qs.append(q)
        ks.append(k)
        vs.append(v)

        flops += (ql * ql ** (1 if causal else 2) + ql * (
                    kvl - ql) * 2) * qo_head * dim * 2

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)

    seg_attn_meta = get_seg_attn_meta(qls, klss, mask=attn_mask)

    flash_attn_meta = get_flash_attn_meta(qls, kls, mask=attn_mask)

    print(f'\nseg:{max_seg} mode:{mode} causal:{causal} bs:{len(qls)} q:{qls[0]} k:{kls[0]} qo_head:{qo_head} kv_head:{kv_head} dim:{dim}')
    n_repeat = 1000
    # org_time = benchmark_func(flash_attn, q, k, v, flash_attn_meta,
    #                         causal=causal, ref_flops=flops,
    #                         n_repeat=n_repeat)
    org_time = None
    benchmark_func(seg_attn, q, k, v, seg_attn_meta, causal=causal,
                n_repeat=n_repeat, ref_time=org_time, ref_flops=flops)


if __name__ == '__main__':
    # for max_seg in [1,2,4]:
    #     for mode in ['prefill', 'decode', 'mix']:
    #         for even in [True, False]:
    #             bench_seg_attn(max_seg=max_seg, mode=mode, even=even, causal=True)
    # for max_seg in [1, 2, 4]:
    #     for even in [True, False]:
    #         bench_seg_attn(max_seg=max_seg, mode='spec', even=even, causal=True)
    bench_seg_attn(max_seg=1, mode='decode', even=True, causal=False)
    # bench_seg_attn(max_seg=2, mode='spec', even=True, causal=True)
