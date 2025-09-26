# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from flood.ops.seg_attn import seg_attn_fwd
from flood.utils.benchmark import *

seed_everything(7)


def seg_attn(q, k, v, meta, online_scale=True):
    outputs = seg_attn_fwd(q, k, v, meta, online_scale=online_scale)

    return outputs


def test_seg_attn(max_seg=1, mode="prefill", even=True, online_scale=True):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    qo_head = 64
    kv_head = 8
    dim = 128
    masks = None
    mask_size = 16
    torch_mask = None
    if mode == "prefill":
        bs = 1
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
    elif mode == "decode":
        bs = 128
        qls = [1] * bs
        if even:
            kls = [1024] * bs
        else:
            kls = [1025] * bs
    elif mode == "mix":
        bs = 40
        qls = [(1024 - bs + 1) // max_seg] + [1] * (bs - 1)
        if even:
            kls = [1024] * bs
        else:
            kls = [1025] * bs
    elif mode == "spec":
        bs = 16
        qls = [mask_size] * bs
        if even:
            kls = [1024] * bs
        else:
            kls = [1025] * bs

        assert all([x == mask_size for x in qls])

        masks = torch.tril(
            torch.ones((bs, mask_size, mask_size), dtype=torch.int8, device=device), 0
        )
        for i in range(mask_size):
            for j in range(mask_size):
                if j < i:
                    masks[:, i, j] = random.randint(0, 1)
        # print(f'{masks}')

        full_mask = torch.ones(
            (bs, qls[0], kls[0] - qls[0]), dtype=torch.float32, device=device
        )
        torch_mask = -10000 * (1 - torch.cat([full_mask, masks.float()], dim=2))
        torch_mask = torch_mask[:, None]
    else:
        raise ValueError(f"unknown mode:{mode}")

    assert all([x <= kls[i] for i, x in enumerate(qls)])
    klss = split(kls, max_seg)
    flops = 0

    # flash_attn = flash_attn_3 if torch.cuda.get_device_properties(
    #         0).major >= 9 else flash_attn_2
    flash_attn = flash_attn_2

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

        flops += (ql * ql + ql * (kvl - ql) * 2) * qo_head * dim * 2

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)

    desc = f"seg:{max_seg} mode:{mode} online:{online_scale} bs:{len(qls)} q:{qls[0]} k:{kls[0]} qo_head:{qo_head} kv_head:{kv_head} dim:{dim}"

    # ks = torch.stack([k]*group,2).view(sum(kls), qo_head, dim)
    # vs = torch.stack([v]*group,2).view(sum(kls), qo_head, dim)

    seg_attn_meta = get_seg_attn_meta(qls, klss, mask=masks)

    flash_attn_meta = get_flash_attn_meta(qls, kls)

    if mode == "spec":
        org_output = torch_attn(
            q.view(bs, mask_size, qo_head, dim),
            k.view(bs, kls[0], kv_head, dim),
            v.view(bs, kls[0], kv_head, dim),
            causal=True,
            mask=torch_mask,
        ).view(bs * qls[0], qo_head, dim)
    else:
        org_output = flash_attn(q, k, v, flash_attn_meta)

    torch.cuda.synchronize()

    opt_output = seg_attn(q, k, v, seg_attn_meta, online_scale=online_scale)
    torch.cuda.synchronize()

    # print(org_output.shape, opt_output.shape)

    # print("org",org_output.dtype,org_output.shape)
    # print("opt",opt_output.dtype,opt_output.shape)
    errs = (opt_output.float() - org_output.float()).abs()
    err = errs.mean().item()
    amp = org_output.float().abs().mean().item()
    rate = err / amp

    print(f"\n{desc} err:{err:.4f} rate:{rate:.3f}")
    if math.isnan(rate) or rate > 0.02:
        print(
            f"org max:{torch.max(org_output).item():.3f} min:{torch.min(org_output).item():.3f}"
        )
        print(
            f"opt max:{torch.max(opt_output).item():.3f} min:{torch.min(opt_output).item():.3f}"
        )

        print(torch.isnan(opt_output).float().argmax())

        print("org_output[:,0,0]", org_output[:, 0, 0])
        print("opt_output[:,0,0]", opt_output[:, 0, 0])

        print("org_output[0,:,0]", org_output[0, :, 0])
        print("opt_output[0,:,0]", opt_output[0, :, 0])

        print("org_output[0,0,:]", org_output[0, 0, :])
        print("opt_output[0,0,:]", opt_output[0, 0, :])
        torch.testing.assert_close(
            opt_output.float(), org_output.float(), rtol=0.05, atol=0.1
        )


if __name__ == "__main__":
    for max_seg in [1, 2, 4]:
        for mode in ["prefill", "decode", "mix", "spec"]:
            for even in [True, False]:
                for online_scale in [True, False]:
                    test_seg_attn(
                        max_seg=max_seg, mode=mode, even=even, online_scale=online_scale
                    )

    # test_seg_attn(max_seg=2, mode='spec', even=False, online_scale=True)
