# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from flood.ops.draft import *
from flood.utils.benchmark import benchmark_func





def test_update_draft(length=8, count=8, batch_size=32, retrieve_count=4, retrieve_length=4):
    size = 2**24
    # length = 8
    # count = 8
    freq_table = torch.zeros((size,), dtype=torch.float32, device="cuda:0")
    draft_table = torch.zeros((size, length), dtype=torch.int32, device="cuda:0")
    tokens = list(range(10000))
    update_draft_table(
        tokens,
        freq_table,
        draft_table,
        table_size=size,
        branch_length=length,
        branch_count=count,
        vocab=128256,
        eos=0,
    )

    # print(freq_table[freq_table>0])
    # print(draft_table[draft_table[:,1]>0])

    # benchmark_func(update_draft_table, tokens, freq_table, draft_table, size=size, length=length, count=count)

    # batch_size = 32
    # retrieve_count = 4
    # retrieve_length = 4
    tokens = [[i * 2, i * 2 + 1] for i in range(batch_size)]
    output_tokens, output_masks = retrieve_draft_table(
        tokens,
        freq_table,
        draft_table,
        table_size=size,
        vocab=128256,
        branch_length=length,
        branch_count=count,
        retrieve_count=retrieve_count,
    )
    print(output_tokens)
    print(output_masks)
    benchmark_func(
        retrieve_draft_table,
        tokens,
        freq_table,
        draft_table,
        size=size,
        length=length,
        count=count,
        retrieve_count=retrieve_count,
        retrieve_length=retrieve_length,
        vocab=100000,
        eos=1000,
    )




def test_update_cache(batch_size=4, n_heads=16, dim=128, branch_length=8, branch_count=1):
    dtype = torch.bfloat16 
    device = 'cuda:0'
    if False:
        cache = torch.zeros((batch_size, n_heads*dim*dim), device=device, dtype=dtype)
        decay_scales = 0.1*torch.rand((n_heads, ), device=device, dtype=torch.float32)
        s_offsets = torch.arange(batch_size, device=device)
        keys = torch.randn((batch_size*branch_length*branch_count, n_heads, dim), device=device, dtype=torch.bfloat16)
        values = torch.randn((batch_size*branch_length*branch_count, n_heads, dim), device=device, dtype=torch.bfloat16)
        tmp = torch.randn((batch_size, branch_length), device=device, dtype=torch.float32)
        accept_indices = torch.argsort(tmp).to(torch.int32)
        accept_indices[:,0:] = -1 
    else:
        ds = torch.load("/tmp/cache.bin")
        cache = ds['cache'].view(-1,n_heads*dim*dim)
        s_offsets = ds['s_offsets']
        keys = ds['ks']
        values = ds['vs']
        accept_indices = ds['accept_indices']
        decay_scales = ds['decay_scales']
        batch_size = s_offsets.shape[0]

        accept_indices[0,:4] = torch.tensor([8,9,10,11],device=device) 

    ref_cache = cache.clone().float().view(-1, n_heads,dim,dim)
    for i in range(batch_size):
        indices = [x for x in accept_indices[i].tolist() if x != -1]
        n_accept = len(indices)
        ref_cache[i] *= torch.exp(-(n_accept+1)*decay_scales[:,None,None])
        for j in range(n_accept+1):
            if j == 0:
                idx = 0 
            else:
                idx = indices[j-1]
            ref_cache[i] += keys[i*branch_length*branch_count+idx,:,:,None] * values[i*branch_length*branch_count+idx,:,None,:] * torch.exp(-(n_accept - j)*decay_scales[:,None,None])

    opt_cache = cache.detach().clone()
    update_draft_fix_size_cache(opt_cache, s_offsets, keys, values, accept_indices, decay_scales)

    opt_cache = opt_cache.view(-1, n_heads,dim,dim)

    rel_err = (opt_cache - ref_cache).abs().sum()/ref_cache.abs().sum()
    print(f'rel:{rel_err.item():3f}')
    # print(f'{cache.view(-1, n_heads,dim,dim)[0,0]=}')
    # print(f'{ref_cache[0,0]=}')
    # print(f'{opt_cache[0,0]=}')

    benchmark_func(update_draft_fix_size_cache, opt_cache.view(-1, n_heads*dim*dim), s_offsets, keys, values, accept_indices, decay_scales)

if __name__ == '__main__':

    # test_update_draft(length=8, count=8, batch_size=32, retrieve_count=4, retrieve_length=4)
    test_update_cache(batch_size=16, n_heads=16, dim=128, branch_length=16, branch_count=4)