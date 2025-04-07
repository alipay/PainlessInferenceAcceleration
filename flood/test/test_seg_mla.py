# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
import torch

from flood.ops.segmla import seg_mla_fwd
from flood.utils.benchmark import benchmark_func

torch.manual_seed(7)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False



def scaled_dot_product_attention_fp32(query, key_value):
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


def scaled_dot_product_attention(query, key_value):
    # query:[bs, q_length, 128, 576]
    # key_vaue: [bs, k_length, 576]
    _, q_length, _, q_dim = query.size()
    k_length = key_value.size(1)
    query = query.clone()
    key = key_value.clone()
    value = key_value[:,:,:512].clone()
    query = query.permute(0,2,1,3)  # [bs, 128, q_length, 576]
    key = key.unsqueeze(1).permute(0,1,3,2)  # [bs, 1, 576, k_length]
    value = value.unsqueeze(1)   # [bs, 1, k_length, 512]
    attn_weight = query @ key / math.sqrt(q_dim)  # [bs, 128, q_length, k_length]
    mask = torch.tril(torch.ones(q_length, k_length, dtype=query.dtype, device=query.device), k_length-q_length)
    # print(mask)
    attn_weight -= 10000*(1-mask)
    lse = torch.exp(attn_weight).sum(-1)
    attn_weight = torch.exp(attn_weight).to(query.dtype)
    # print(attn_weight[0,0,1,:4])
    output = attn_weight @ value  # [bs, 128, q_length, 512]
    output = output/lse[...,None]
    lse = lse.permute(0,2,1)
    output = output.permute(0,2,1,3).contiguous()
    return output, lse


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


def get_seg_mla_meta(qls, klss, mask=None):
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




def test_seg_mla(max_seg=1, mode='prefill', even=True):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
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
    else:
        raise ValueError(f'unknown mode:{mode}')

    klss = split(kls, max_seg)
    flops = 0

    attn_mask = None

    qs = []
    kvs = []
    flops = 0.0
    for i, ql in enumerate(qls):
        kvl = kls[i]
        q = torch.randn(ql, 128, 576, dtype=dtype, device=device)
        kv = torch.randn(kvl, 576, dtype=dtype, device=device)
        qs.append(q)
        kvs.append(kv)

        flops += (ql * ql + ql * (
                    kvl - ql) * 2) * 128 * 576 * 2

    q = torch.cat(qs, 0)
    kv = torch.cat(kvs, 0)

    seg_attn_meta = get_seg_mla_meta(qls, klss, mask=attn_mask)

    org_outputs = []
    for i, qi in enumerate(qs):
        kvi = kvs[i]
        org_output, org_lse = scaled_dot_product_attention_fp32(qi[None], kvi[None])
        org_outputs.append(org_output[0])
    org_output = torch.cat(org_outputs, 0)
    

    torch.cuda.synchronize()

    opt_output = seg_mla_fwd(q, kv, seg_attn_meta)
    torch.cuda.synchronize()

    # print("org",org_output.dtype,org_output.shape)
    # print("opt",opt_output.dtype,opt_output.shape)
    errs = (opt_output.float() - org_output.float()).abs()
    err = errs.mean().item()
    amp = org_output.float().abs().mean().item()
    rate = err / amp

    desc = f'seg:{max_seg} mode:{mode} bs:{len(qls)} q:{qls[0]} k:{kls[0]}'

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
    test_seg_mla(max_seg=1, mode='prefill', even=True)
    # for max_seg in [1,2,4]:
    #     for mode in ['prefill', 'decode', 'mix']:
    #         for even in [True, False]:
    #             test_seg_mla(max_seg=max_seg, mode=mode, even=even)

