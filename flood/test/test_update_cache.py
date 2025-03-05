# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from flood.ops import update_cache

if __name__ == '__main__':
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    ql = 427
    q_head = 28
    kv_head = 4
    dim = 128
    qkv = torch.randn(ql, q_head + 2 * kv_head, dim, dtype=dtype, device=device)
    k = qkv[:, q_head:q_head + kv_head]
    v = qkv[:, q_head + kv_head:]
    # print(f'{k.shape=} {v.shape=}')
    ks = torch.randn(ql, kv_head, dim, dtype=dtype, device=device)
    vs = torch.randn(ql, kv_head, dim, dtype=dtype, device=device)
    indices = list(range(ql))
    random.shuffle(indices)
    indices = torch.tensor(indices, device=device, dtype=torch.int32)
    update_cache(ks, vs, k, v, indices)

    k_float = k.float()
    v_float = v.float()

    ks_float = ks.float()[indices]
    vs_float = vs.float()[indices]

    # print('org',k_float[0,0])
    # print('opt',ks_float[0,0])

    torch.testing.assert_close(k_float, ks_float, rtol=0.01, atol=0.01)
    torch.testing.assert_close(v_float, vs_float, rtol=0.01, atol=0.01)
