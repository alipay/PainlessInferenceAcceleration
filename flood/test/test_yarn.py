# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import flood_cuda
# from torch._prims_common import Dim
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding, \
    apply_rotary_pos_emb_interleave, DeepseekV3Config

from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding, \
    apply_rotary_emb as ds_v2_apply_rotary_emb, DeepseekV2Config


from flood.layers.rope import DeepseekYarnRope, YarnRope
from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIABLE_DEVICES'] = '0'

if __name__ == '__main__':
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    # dtype = torch.float16

    bs = 2
    ql = 10
    kvl = 0
    dim = 64
    qo_head = 128
    kv_head = 128
    qk_rope_head_dim = 64
    rope_theta = 10000
    q = torch.rand(bs, ql, qo_head, dim).to(dtype=dtype, device=device)
    k = torch.rand(bs, ql, 1, dim).to(dtype=dtype, device=device)
    q_rot = q.transpose(1, 2)
    k_rot = k.transpose(1, 2)
    position_ids = torch.stack([torch.arange(kvl, kvl + ql)] * bs, 0).to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    rope_scaling = {
      "beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0, "mscale_all_dim": 1.0, 
    "original_max_position_embeddings": 4096, "rope_type": "yarn", "type": "yarn"
    }
    conf = DeepseekV3Config(hidden_size=qo_head*dim, num_attention_heads=qo_head,rope_theta=rope_theta,max_position_embeddings=163840, rope_scaling=rope_scaling)
    rotary_emb = DeepseekV3RotaryEmbedding(conf, device=device)
    cos, sin = rotary_emb(q, position_ids)
    cos_ref, sin_ref = cos[..., :32], sin[..., :32]
    q_rope_ref, k_rope_ref = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

    indptr = torch.tensor(
        [i * ql for i in range(bs + 1)], dtype=torch.int32, device=device
    )
    offsets = torch.full((bs,), kvl, dtype=torch.int32, device=device)

    q_clone = q.clone().view(bs * ql, qo_head, dim)
    k_clone = k.clone().view(bs * ql, 1, dim)

    positions = []
    for i, pos in enumerate(offsets):
        positions.append(list(range(pos, pos+indptr[i+1]-indptr[i])))
    positions = torch.tensor(positions, device=device)
    positions = positions.flatten()

    assert all(position_ids.flatten() == positions)

    # extra_kwargs = {
    # k: v
    # for k, v in rope_scaling.items()
    # if k in ("extrapolation_factor", "attn_factor", "beta_fast",
    #             "beta_slow", "mscale", "mscale_all_dim")}

    # vllm_rope = DeepseekScalingRotaryEmbedding(dim, dim, 4096, rope_theta, False, 40, torch.bfloat16, **extra_kwargs)
    # q_vllm, k_vllm = vllm_rope(positions, q_clone, k_clone)

    # q_vllm = q_vllm.view(bs, ql, qo_head, dim).transpose(1, 2)
    # k_vllm = k_vllm.view(bs, ql, 1, dim).transpose(1, 2)

    ds_v3_roep = DeepseekYarnRope(conf)

    cos_sin = ds_v3_roep.cos_sin_cache.to(q.device).index_select(0, positions)
    cos_ds, sin_ds = cos_sin.chunk(2, dim=-1)
    cos_ds = cos_ds.view(bs, ql, -1)
    sin_ds = sin_ds.view(bs, ql, -1)
    
    q_ds, k_ds = ds_v3_roep(q_clone, k_clone, indptr, offsets)
    q_ds = q_ds.view(bs, ql, qo_head, dim).transpose(1, 2)
    k_ds = k_ds.view(bs, ql, 1, dim).transpose(1, 2)


    flood_yarn = YarnRope(conf)
    q_ker, k_ker = flood_yarn(q_clone, k_clone, indptr, offsets)
    q_ker = q_ker.view(bs, ql, qo_head, dim).transpose(1, 2)
    k_ker = k_ker.view(bs, ql, 1, dim).transpose(1, 2)

    torch.testing.assert_close(q_ds, q_ker, rtol=0.1, atol=0.1)
    torch.testing.assert_close(k_ds, k_ker, rtol=0.1, atol=0.1)
    