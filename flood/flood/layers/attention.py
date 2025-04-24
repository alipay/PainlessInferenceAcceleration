# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import math
import torch

from flood.ops.seg_attn import seg_attn_fwd
from flood.ops.seg_mla import seg_mla_fwd


try:
    import flash_attn_2_cuda
except:
    flash_attn_2_cuda = None

try:
    import flash_attn_3_cuda
except:
    flash_attn_3_cuda = None
    print("flash_attn_3_cuda not found!")

from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.chunk import chunk_simple_gla


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
            if 'fla' in kernels:
                return Fp16LinearAttention(layer_idx)
            elif 'fa3' in kernels:
                return Fp16Attention3(layer_idx, softmax_scale=softmax_scale)
            elif 'fa2' in kernels:
                return Fp16Attention(layer_idx, softmax_scale=softmax_scale)
            elif 'mla' in kernels:
                return Fp16SegMla(layer_idx, softmax_scale=softmax_scale)
            else:
                return Fp16SegAttention(layer_idx, softmax_scale=softmax_scale)
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
        key_states, value_states = cache.update_cache(key_states, value_states,
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
    def __init__(self, layer_idx, softmax_scale=1.0, online_scale=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.online_scale = online_scale

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        # if any([x.done>0 for x in batch_meta_info.reqs]):
        #     print('debug') 
        key_states, value_states = cache.update_cache(key_states, 
                                                      value_states,
                                                      self.layer_idx, 
                                                      batch_meta_info)
        # batch_meta_info.max_seg = 1
        # batch_meta_info.mask = None
        output = seg_attn_fwd(
            query_states,
            key_states,
            value_states,
            batch_meta_info,
            online_scale=self.online_scale
        )

        return output

class Fp16LinearAttention(torch.nn.Module):
    def __init__(self, layer_idx, linear_scale=None, linear_mode='chunk'):
        super().__init__()
        self.layer_idx = layer_idx
        self.lightning_attn_ops = {
            'fused_recurrent': fused_recurrent_simple_gla,
            'chunk': chunk_simple_gla
        }
        self.linear_scale = linear_scale
        self.linear_mode = linear_mode

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        # if any([x.done>0 for x in batch_meta_info.reqs]):
        #     print('debug') 
        past_key_value_states = cache.caches[self.layer_idx]
        H = query_states.shape[2]
        s = -self.build_slope_tensor(H) * (1 - self.layer_idx / (self.num_layers - 1) + 1e-5)
        g = s[None, None, :].expand(query_states.shape[0], query_states.shape[1], query_states.shape[2]).contiguous()

        if self.mode in self.lightning_attn_ops:
            output, past_key_value_states = self.lightning_attn_ops[self.mode](
                q=query_states,
                k=key_states,
                v=value_states,
                g=g,
                scale=self.linear_scale,
                initial_state=past_key_value_states,
                output_final_state=True,
                head_first=False
            )
        else:
            raise NotImplementedError(f"Not supported mode `{self.mode}`.")

        return output

    def build_slope_tensor(self, num_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n)  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2)
                        + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = torch.tensor(get_slopes(num_attention_heads), dtype=torch.float)
        return slopes


class Fp16SegMla(torch.nn.Module):
    def __init__(self, layer_idx, softmax_scale=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale

    def forward(self, query_states, key_value_states, batch_meta_info,
                cache):
        key_value_states = cache.update_fusion_cache(key_value_states,
                                                self.layer_idx, batch_meta_info)
        output = seg_mla_fwd(
            query_states,
            key_value_states,
            batch_meta_info
        )

        return output




class Fp16Attention3(torch.nn.Module):
    def __init__(self, layer_idx, softmax_scale=1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale

    def forward(self, query_states, key_states, value_states, batch_meta_info,
                cache):
        key_states, value_states = cache.update_cache(key_states, value_states,
                                                self.layer_idx, batch_meta_info)

        # outputs = flashattn_hopper_cuda.varlen_fwd(
        #     query_states,
        #     key_states,
        #     value_states,
        #     None,  # out_
        #     batch_meta_info.q_offsets,
        #     batch_meta_info.k_offsets,
        #     None,  # meta.seqused_q,
        #     batch_meta_info.k_lengths,
        #     batch_meta_info.max_q_length,
        #     batch_meta_info.max_k_length,
        #     self.softmax_scale,
        #     True,  # causal 
        #     -1,  # window_size_left
        #     -1,  # window_size_right
        # )
        causal = True
        pack_gqa = True
        outputs = flash_attn_3_cuda.fwd(
                query_states,
                key_states,
                value_states,
                None,
                None,
                None,
                None,
                batch_meta_info.q_offsets,
                batch_meta_info.k_offsets,
                None,
                batch_meta_info.q_lengths,
                batch_meta_info.k_lengths,
                batch_meta_info.max_q_length,
                batch_meta_info.max_k_length,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                self.softmax_scale,
                causal,
                -1,
                -1,
                0.0,
                False,
                0,
                pack_gqa,
                0
            )
        return outputs[0]

