# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from flood.ops.segattn import seg_attn_fwd

try:
    import flash_attn_2_cuda
except:
    flash_attn_2_cuda = None
    print("flash_attn_2_cuda not found!")

try:
    import flash_attn_ada_cuda
except:
    flash_attn_ada_cuda = None
    print("flash_attn_ada_cuda not found!")

try:
    import flashattn_hopper_cuda
except:
    flashattn_hopper_cuda = None
    print("flashattn_hopper_cuda not found!")


class AutoAttention():

    @classmethod
    def from_pretrained(cls,
                        dtype,
                        layer_idx=0,
                        softmax_scale=1.0,
                        kernels=('sa',),
                        name=None):
        if layer_idx == 0:
            print(f"attention dtype:{dtype}")
        if dtype is None or dtype == torch.bfloat16 or dtype == torch.float16 or dtype in (
        'float16', 'bfloat16'):
            if 'fa3' in kernels:
                return Fp16Attention3(layer_idx, softmax_scale=softmax_scale)
            elif 'fa2' in kernels:
                return Fp16Attention(layer_idx, softmax_scale=softmax_scale)
            else:
                return Fp16SegAttention(layer_idx, softmax_scale=softmax_scale)
        elif dtype == torch.float8_e4m3fn or dtype in ('float8_e4m3fn',):
            return Fp8Attention(layer_idx, softmax_scale=softmax_scale)
        else:
            raise ValueError(f'unknown dtype:{dtype}')

    @staticmethod 
    def interleave(query_key_value, num_heads, num_key_value_heads, head_dim):
        permute = []
        for g in range(num_key_value_heads):
            offset = (
                                    num_heads + num_key_value_heads + g) * head_dim
            for i in range(head_dim // 16):
                for j in range(8):
                    permute.append(offset + i * 16 + j)
                    permute.append(offset + i * 16 + j + 8)
        permute = torch.tensor(permute, dtype=torch.int32,
                                device=query_key_value.weight.data.device)
        offset = (num_heads + num_key_value_heads) * head_dim
        if query_key_value.weight.data.dtype == torch.float8_e4m3fn:
            query_key_value.weight.data.view(torch.int8)[:,
            offset:] = query_key_value.weight.data.view(torch.int8)[:,
                        permute]
        else:
            query_key_value.weight.data[offset:] = \
            query_key_value.weight.data[permute]
        if query_key_value.bias is not None:
            query_key_value.bias.data[offset:] = \
            query_key_value.bias.data[permute]


class Fp16Attention(torch.nn.Module):
    def __init__(self, layer_idx, softmax_scale=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        key_states, value_states = cache.update(key_states, value_states,
                                                self.layer_idx, batch_meta_info)

        outputs = flash_attn_2_cuda.varlen_fwd(
            query_states,
            key_states,
            value_states,
            None,  # out_
            batch_meta_info.q_offsets,
            batch_meta_info.k_offsets,
            batch_meta_info.k_lengths,
            None,  # leftpad_k
            None,  # block_table
            None,  # alibi_slopes
            batch_meta_info.max_q_length,
            batch_meta_info.max_k_length,
            0.0,  # dropout
            self.softmax_scale,
            False,  # zero_tensors
            True,  # causal 
            -1,  # window_size_left
            -1,  # window_size_right
            0.0,  # softcap
            False,  # return_softmax
            None  # Generator
        )

        return outputs[0]


class Fp16SegAttention(torch.nn.Module):
    def __init__(self, layer_idx, softmax_scale=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        key_states, value_states = cache.update(key_states, value_states,
                                                self.layer_idx, batch_meta_info)
        # batch_meta_info.max_seg = 1
        # batch_meta_info.mask = None
        output = seg_attn_fwd(
            query_states,
            key_states,
            value_states,
            batch_meta_info,
            causal=True
        )

        return output


class Fp16Attention3(torch.nn.Module):
    def __init__(self, layer_idx, softmax_scale=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        key_states, value_states = cache.update(key_states, value_states,
                                                self.layer_idx, batch_meta_info)

        outputs = flashattn_hopper_cuda.varlen_fwd(
            query_states,
            key_states,
            value_states,
            None,  # out_
            batch_meta_info.q_offsets,
            batch_meta_info.k_offsets,
            None,  # meta.seqused_q,
            batch_meta_info.k_lengths,
            batch_meta_info.max_q_length,
            batch_meta_info.max_k_length,
            self.softmax_scale,
            True,  # causal 
            -1,  # window_size_left
            -1,  # window_size_right
        )
        return outputs[0]


class Fp8Attention(torch.nn.Module):
    def __init__(self, layer_idx, softmax_scale=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.query_scale = None
        self.key_scale = None
        self.value_scale = None

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        # if self.value_scale is None:
        #     time.sleep(0.01)
        #     self.query_scale = 2.0**int(math.log2(0.25*448.0/query_states.abs().max().float().item()))
        #     time.sleep(0.01)
        #     self.key_scale = 2.0**int(math.log2(0.25*448.0/key_states.abs().max().float().item()))
        #     time.sleep(0.01)
        #     self.value_scale = 2.0**int(math.log2(0.25*448.0/value_states.abs().max().float().item()))
        #     # self.softmax_scale /= self.query_scale*self.key_scale
        #     time.sleep(0.01)
        #     # print(f'layer:{self.layer_idx} query_scale:{self.query_scale:.3f} key_scale:{self.key_scale:.3f} value_scale:{self.value_scale:.3f}')

        # query_states *= self.query_scale
        # key_states *= self.key_scale
        # value_states *= self.value_scale
        query_states, key_states, value_states = cache.quant_to_fp8_and_update(
            query_states, key_states, value_states, self.layer_idx,
            batch_meta_info)

        outputs = flash_attn_ada_cuda.varlen_fwd(
            query_states,
            key_states,
            value_states,
            None,  # out
            batch_meta_info.q_offsets,
            batch_meta_info.k_offsets,
            batch_meta_info.k_lengths,
            None,  # leftpad_k
            None,  # block_table
            None,  # alibi_slopes
            batch_meta_info.max_q_length,
            batch_meta_info.max_k_length,
            0.0,  # dropout
            self.softmax_scale,
            False,  # zero_tensors
            True,  # causal 
            -1,  # window_size_left
            -1,  # window_size_right
            0.0,  # softcap
            False,  # return_softmax
            None  # Generator
        )

        return outputs[0]
        # return outputs[0]/self.value_scale
