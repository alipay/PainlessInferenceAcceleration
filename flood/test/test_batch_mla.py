# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
import torch

from flood.ops.batchmla import batch_mla_fwd
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



if __name__ == '__main__':
    device = 'cuda:0'
    dtype = torch.bfloat16 
    bs = 1
    q_length = 4095
    k_length = 4096
    q = torch.randn((bs,q_length,128,576), device=device, dtype=dtype)
    kv = torch.randn((bs,k_length,576), device=device, dtype=dtype)

    ref_output, ref_lse = scaled_dot_product_attention(q, kv)
    ref_output = ref_output.float()

    opt_output, opt_lse = batch_mla_fwd(q, kv)
    opt_output = opt_output.float()

    # print(f'{ref_output.shape=} {opt_output.shape=}')

    output_err = ((ref_output-opt_output).abs().mean()/ref_output.abs().mean()).item()
    lse_err = ((ref_lse-opt_lse).abs().mean()/ref_lse.abs().mean()).item()

    print(f"\noutput_err:{output_err:.3f} lse_err:{lse_err:.3f}\n")
    # print(f'\n{ref_output[0,0,0,:4]=}') 
    # print(f'{opt_output[0,0,0,:4]=}')

    # for i in range(1,10):
    #     print(f'\n{ref_output[0,i,0,:4]=}') 
    #     print(f'{opt_output[0,i,0,:4]=}')

    # print(f'\n{ref_output[0,-1,0,:4]=}') 
    # print(f'{opt_output[0,-1,0,:4]=}')

    # print(f'\n{ref_output[-1,0,0,:4]=}') 
    # print(f'{opt_output[-1,0,0,:4]=}\n')

    benchmark_func(batch_mla_fwd, q, kv, n_repeat=100, ref_flops=bs*q_length*k_length*128*(576+512)*2/2)