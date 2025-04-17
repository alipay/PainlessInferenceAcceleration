# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from flood.ops.seg_attn import seg_attn_fwd
from flood.utils.benchmark import *


def seg_attn(q, k, v, meta, online_scale=True):
    outputs = seg_attn_fwd(
        q,
        k,
        v,
        meta,
        online_scale=online_scale
    )

    return outputs

def bench_seg_attn(max_seg=1, mode='prefill', even=True, online_scale=True):
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

    # flash_attn = flash_attn_3 if torch.cuda.get_device_properties(
    #         0).major >= 9 else flash_attn_2
    flash_attn = flash_attn_2

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

        flops += (ql * ql * 1 + ql * (
                    kvl - ql) * 2) * qo_head * dim * 2

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)

    seg_attn_meta = get_seg_attn_meta(qls, klss, mask=attn_mask)

    flash_attn_meta = get_flash_attn_meta(qls, kls, mask=attn_mask)

    print(f'\nseg:{max_seg} mode:{mode} bs:{len(qls)} q:{qls[0]} k:{kls[0]} qo_head:{qo_head} kv_head:{kv_head} dim:{dim}')
    n_repeat = 1000
    org_time = benchmark_func(flash_attn, q, k, v, flash_attn_meta,
                            ref_flops=flops,
                            n_repeat=n_repeat)
    benchmark_func(seg_attn, q, k, v, seg_attn_meta, online_scale=online_scale,
                n_repeat=n_repeat, ref_time=org_time, ref_flops=flops)


if __name__ == '__main__':
    for max_seg in [1]:
        for mode in ['prefill', 'decode', 'mix']:
            for even in [True, False]:
                for online_scale in [False, True]:
                    bench_seg_attn(max_seg=max_seg, mode=mode, even=even, online_scale=online_scale)

    # bench_seg_attn(max_seg=1, mode='decode', even=True)
    # bench_seg_attn(max_seg=2, mode='spec', even=True)
