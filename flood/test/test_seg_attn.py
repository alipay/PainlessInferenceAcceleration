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



def test_seg_attn(max_seg=1, mode='prefill', even=True, online_scale=True):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    qo_head = 32
    kv_head = 8
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

    # flash_attn = flash_attn_3 if torch.cuda.get_device_properties(
    #         0).major >= 9 else flash_attn_2
    flash_attn = flash_attn_2
    if mask:
        attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.int8,
                                device=device)
        # attn_mask = torch.zeros((mask_size,mask_size),dtype=torch.float32, device=device)
        for i in range(mask_size):
            attn_mask[i, i + 1:] = 1
            # if i > 1:
            #     attn_mask[i,1] = 1
        # attn_mask[:] = 1
        # print(f'{attn_mask=}')
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

        flops += (ql * ql + ql * (
                    kvl - ql) * 2) * qo_head * dim * 2

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)

    # ks = torch.stack([k]*group,2).view(sum(kls), qo_head, dim)
    # vs = torch.stack([v]*group,2).view(sum(kls), qo_head, dim)

    seg_attn_meta = get_seg_attn_meta(qls, klss, mask=attn_mask)

    flash_attn_meta = get_flash_attn_meta(qls, kls, mask=attn_mask)

    if mask:
        assert qls[0] == mask_size and len(qls) == 1
        torch_mask = -10000 * torch.cat([torch.zeros((qls[0], kls[0] - qls[0]),
                                                     dtype=torch.float32,
                                                     device=device),
                                         attn_mask.float()], dim=1)
        org_output = \
        torch_attn(q[None], k[None], v[None], causal=True, mask=torch_mask)[0]
    else:
        org_output = flash_attn(q, k, v, flash_attn_meta)
    torch.cuda.synchronize()

    opt_output = seg_attn(q, k, v, seg_attn_meta, online_scale=online_scale)
    torch.cuda.synchronize()

    # print("org",org_output.dtype,org_output.shape)
    # print("opt",opt_output.dtype,opt_output.shape)
    errs = (opt_output.float() - org_output.float()).abs()
    err = errs.mean().item()
    amp = org_output.float().abs().mean().item()
    rate = err / amp

    desc = f'seg:{max_seg} mode:{mode} bs:{len(qls)} q:{qls[0]} k:{kls[0]} qo_head:{qo_head} kv_head:{kv_head} dim:{dim}'

    print(f"\n{desc} err:{err:.4f} rate:{rate:.3f}")
    if math.isnan(rate) or rate > 0.02:
        print(
            f"org max:{torch.max(org_output).item():.3f} min:{torch.min(org_output).item():.3f}")
        print(
            f"opt max:{torch.max(opt_output).item():.3f} min:{torch.min(opt_output).item():.3f}")

        print(torch.isnan(opt_output).float().argmax())

        print("org_output[:,0,0]", org_output[:, 0, 0])
        print("opt_output[:,0,0]", opt_output[:, 0, 0])

        print("org_output[0,:,0]", org_output[0, :, 0])
        print("opt_output[0,:,0]", opt_output[0, :, 0])

        print("org_output[0,0,:]", org_output[0, 0, :])
        print("opt_output[0,0,:]", opt_output[0, 0, :])
        torch.testing.assert_close(opt_output.float(), org_output.float(),
                                   rtol=0.05, atol=0.1)

if __name__ == '__main__':
    # for max_seg in [1,2,4]:
    #     for mode in ['prefill', 'decode', 'mix']:
    #         for even in [True, False]:
    #             test_seg_attn(max_seg=max_seg, mode=mode, even=even)

    test_seg_attn(max_seg=2, mode='decode', even=False, online_scale=True)
