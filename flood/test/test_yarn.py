# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import math
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


class NativeDeepseekYarnRope(torch.nn.Module):
    def __init__(self, config, is_neox_style=True):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.scaling_factor = config.rope_scaling.get('factor', 4.0)
        self.beta_slow = config.rope_scaling.get('beta_slow', 1.0)
        self.beta_fast = config.rope_scaling.get('beta_fast', 32.0)
        self.mscale = config.rope_scaling.get('mscale', 1.0)
        self.mscale_all_dim = config.rope_scaling.get('mscale_all_dim', 1.0)
        self.mscale = float(
            self.yarn_get_mscale(self.scaling_factor, float(self.mscale)) /
            self.yarn_get_mscale(self.scaling_factor, float(self.mscale_all_dim)))
        self.original_max_position_embeddings = config.rope_scaling.get(
            'original_max_position_embeddings', 16384)
        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, 'partial_rotary_factor') else 1.0
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        self.low, self.high = self._yarn_find_correction_range(self.beta_fast,
                                                               self.beta_slow,
                                                               self.rotary_dim,
                                                               self.rope_theta,
                                                               self.original_max_position_embeddings)
        cache = self._compute_cos_sin_cache()
        cache = cache.to(torch.bfloat16)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self.is_neox_style = is_neox_style

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.original_max_position_embeddings * self.scaling_factor,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.rope_theta**(
            torch.arange(0,
                         self.rotary_dim,
                         2,
                         dtype=torch.float) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - self._yarn_linear_ramp_mask(
            self.low, self.high, self.rotary_dim // 2,
            dtype=torch.float))
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _yarn_find_correction_dim(self,
                                  num_rotations: int,
                                  dim: int,
                                  base: float,
                                  original_max_position_embeddings: int = 2048):
        return (dim * math.log(original_max_position_embeddings /
                               (num_rotations * 2 * math.pi))) / (
                    2 * math.log(base))

    def _yarn_find_correction_range(self,
                                    low_rot: int,
                                    high_rot: int,
                                    dim: int,
                                    base: float,
                                    max_position_embeddings: int = 2048):
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base,
                                           max_position_embeddings))
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base,
                                           max_position_embeddings))
        return float(max(low, 0)), float(min(high, dim - 1))

    def _yarn_linear_ramp_mask(self, low: float, high: float, dim: int,
                            dtype: torch.dtype) -> torch.Tensor:
        if low == high:
            high += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def rotate_gptj(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    # common functions
    def rotate_neox(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 indptr: torch.Tensor,
                 offsets: torch.Tensor):
        query_rot = q[..., :self.rotary_dim]
        key_rot = k[..., :self.rotary_dim]
        if self.rotary_dim < self.head_dim:
            query_pass = q[..., self.rotary_dim:]
            key_pass = k[..., self.rotary_dim:]

        positions = []
        for i, pos in enumerate(offsets):
            positions.append(list(range(pos, pos+indptr[i+1]-indptr[i])))
        positions = torch.tensor(positions, device=q.device)
        positions = positions.flatten()
        self.cos_sin_cache = self.cos_sin_cache.to(q.device)
        cos_sin = self.cos_sin_cache.index_select(0, positions).to(q.device)
        cos, sin = cos_sin.chunk(2, dim=-1)

        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [seq_len, rotray_dim // 2].
            cos = cos.repeat(1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = self.rotate_neox if self.is_neox_style else self.rotate_gptj

        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_dim:
            q = torch.cat((query_rot, query_pass), dim=-1)
            k = torch.cat((key_rot, key_pass), dim=-1)
        else:
            q = query_rot
            k = key_rot
        return q, k.squeeze()

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

    native_rope = NativeDeepseekYarnRope(conf, is_neox_style=False)
    
    q_ref, k_ref = native_rope(q_clone, k_clone, indptr, offsets)
    q_ref = q_ref.view(bs, ql, qo_head, dim)
    k_ref = k_ref.view(bs, ql, 1, dim)


    q_clone = q.clone().view(bs * ql, qo_head, dim)
    k_clone = k.clone().view(bs * ql, 1, dim)

    flood_yarn = DeepseekYarnRope(conf)
    q_ker, k_ker = flood_yarn(q_clone, k_clone, indptr, offsets)
    q_ker = q_ker.view(bs, ql, qo_head, dim)
    k_ker = k_ker.view(bs, ql, 1, dim)

    torch.testing.assert_close(q_ref, q_ker, rtol=0.01, atol=0.01)
    torch.testing.assert_close(k_ref, k_ker, rtol=0.01, atol=0.01)
    