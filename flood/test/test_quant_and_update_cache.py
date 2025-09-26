# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from flood.ops import quant_and_update_cache

if __name__ == "__main__":
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    fp8 = torch.float8_e4m3fn

    ql = 1234
    qo_head = 28
    kv_head = 4
    dim = 128
    qkv = torch.randn(ql, qo_head + 2 * kv_head, dim, dtype=dtype, device=device)
    q = qkv[:, :qo_head]
    k = qkv[:, qo_head : qo_head + kv_head]
    v = qkv[:, qo_head + kv_head :]
    # dummy = torch.arange(ql, dtype=torch.int32, device=device).view(-1,1,1).to(dtype)
    # k[:] = dummy
    # v[:] = dummy

    q8 = q.to(fp8)
    k8 = k.to(fp8)
    v8 = v.to(fp8)

    qq = torch.empty_like(q8)
    kq = torch.empty_like(k8)
    vq = torch.empty_like(v8)
    # print(f'{qq.stride()=} {kq.stride()=} {vq.stride()=}')
    indices = list(range(ql))
    random.shuffle(indices)
    indices = torch.tensor(indices, device=device, dtype=torch.int32)
    quant_and_update_cache(qq, kq, vq, q, k, v, indices)
    # torch.cuda.synchronize()

    # print('indices',indices)
    q8_float = q8.float()
    k8_float = k8.float()
    v8_float = v8.float()

    qq_float = qq.float()
    kq_float = kq.float()[indices]
    vq_float = vq.float()[indices]

    # print('org',q8_float[0,0])
    # print('opt',qq_float[0,0])

    # print('org',k8_float[0,0])
    # print('opt',kq_float[0,0])

    # print('org',v8_float[0,0])
    # print('opt',vq_float[0,0])

    torch.testing.assert_close(q8_float, qq_float, rtol=0.01, atol=0.01)
    torch.testing.assert_close(k8_float, kq_float, rtol=0.01, atol=0.01)
    torch.testing.assert_close(v8_float, vq_float, rtol=0.01, atol=0.01)
