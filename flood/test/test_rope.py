# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import flood_cuda

# from torch._prims_common import Dim
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    LlamaConfig,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXRotaryEmbedding,
    apply_rotary_pos_emb as gpt_neox_apply_rotary_pos_emb,
    GPTNeoXConfig,
)


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIABLE_DEVICES'] = '0'

if __name__ == "__main__":
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # dtype = torch.float16

    bs = 2
    ql = 10
    kvl = 0
    dim = 128
    qo_head = 16
    kv_head = 4
    rope_theta = 10000
    q = torch.rand(bs, ql, qo_head, dim).to(dtype=dtype, device=device)
    k = torch.rand(bs, ql, kv_head, dim).to(dtype=dtype, device=device)
    position_ids = torch.stack([torch.arange(kvl, kvl + ql)] * bs, 0).to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    conf = LlamaConfig(
        hidden_size=qo_head * dim,
        num_attention_heads=qo_head,
        rope_theta=rope_theta,
        max_position_embeddings=4096,
    )
    rotary_emb = LlamaRotaryEmbedding(conf, device=device)
    cos, sin = rotary_emb(q, position_ids)
    q_rope_ref, k_rope_ref = llama_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

    q_rope_ref = q_rope_ref.reshape(bs, ql, qo_head, dim)
    k_rope_ref = k_rope_ref.reshape(bs, ql, kv_head, dim)

    indptr = torch.tensor(
        [i * ql for i in range(bs + 1)], dtype=torch.int32, device=device
    )
    offsets = torch.full((bs,), kvl, dtype=torch.int32, device=device)

    q_clone = q.clone().view(bs * ql, qo_head, dim)
    k_clone = k.clone().view(bs * ql, kv_head, dim)

    flood_cuda.apply_rope_inplace(
        q_clone, k_clone, indptr, offsets, dim, False, 1.0, rope_theta
    )

    q_ker = q_clone.view(bs, ql, qo_head, dim)
    k_ker = k_clone.view(bs, ql, kv_head, dim)

    torch.testing.assert_close(q_ker, q_rope_ref, rtol=0.01, atol=0.01)
    torch.testing.assert_close(k_ker, k_rope_ref, rtol=0.01, atol=0.01)

    # partial rope
    partial_rotary_factor = 0.5
    rotary_dim = int(partial_rotary_factor * dim)
    conf = GPTNeoXConfig(
        hidden_size=qo_head * dim,
        num_attention_heads=qo_head,
        rope_theta=rope_theta,
        max_position_embeddings=4096,
        rotary_pct=partial_rotary_factor,
    )
    rotary_emb = GPTNeoXRotaryEmbedding(conf, device=device)
    cos, sin = rotary_emb(q, position_ids)
    q_gpt_ref, k_gpt_ref = gpt_neox_apply_rotary_pos_emb(
        q, k, cos, sin, unsqueeze_dim=2
    )
    q_gpt_ref = q_gpt_ref.reshape(bs, ql, qo_head, dim)
    k_gpt_ref = k_gpt_ref.reshape(bs, ql, kv_head, dim)

    query_flood_inplace = q.clone().view(bs * ql, qo_head, dim)
    key_flood_inplace = k.clone().view(bs * ql, kv_head, dim)
    flood_cuda.apply_rope_inplace(
        query_flood_inplace,
        key_flood_inplace,
        indptr,
        offsets,
        rotary_dim,
        False,
        1.0,
        rope_theta,
    )
    query_flood_inplace = query_flood_inplace.view(bs, ql, qo_head, dim)
    key_flood_inplace = key_flood_inplace.view(bs, ql, kv_head, dim)

    torch.testing.assert_close(q_gpt_ref, query_flood_inplace, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_gpt_ref, key_flood_inplace, atol=1e-2, rtol=1e-2)
