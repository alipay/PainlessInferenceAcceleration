# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import flood_cuda
import torch


class AutoRope(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, config):
        if not hasattr(config,
                       'rope_scaling') or config.rope_scaling is None or isinstance(
                config.rope_scaling, (int, float)):
            return NativeRope(config)
        assert isinstance(config.rope_scaling, dict)
        rope_type = config.rope_scaling.get(
            'rope_type') or config.rope_scaling.get('type')
        if rope_type == 'yarn':
            if config.model_type == 'deepseek_v2' or config.model_type == 'deepseek_v3':
                return DeepseekYarnRope(config)
            else:
                return YarnRope(config)
        elif rope_type == 'llama3':
            return Llama31Rope(config)
        else:
            raise ValueError(f'unknown rope type:{rope_type}')


class NativeRope(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rope_scale = config.rope_scaling if hasattr(config,
                                                         'rope_scaling') and config.rope_scaling is not None else 1.0
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, 'partial_rotary_factor') else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.rotary_dim = int(head_dim * partial_rotary_factor)


    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 indptr: torch.Tensor,
                 offsets: torch.Tensor):
        flood_cuda.apply_rope_inplace(q, k, indptr, offsets, self.rotary_dim, False,
                                      self.rope_scale, self.rope_theta)
        return q, k


class YarnRope(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rope_scale = config.rope_scaling.get('factor', 4.0)
        self.beta_slow = config.rope_scaling.get('beta_slow', 1.0)
        self.beta_fast = config.rope_scaling.get('beta_fast', 32.0)
        self.mscale = config.rope_scaling.get('mscale', 1.0)
        self.mscale_all_dim = config.rope_scaling.get('mscale_all_dim', 1.0)
        # self.head_dim = config.rope_scaling.get('beta_fast', 32.0)
        # self.magnitude_scale = float(
        #     0.1 * math.log(config.rope_scaling.get('factor', 4.0)) + 1.0)
        self.magnitude_scale = float(
            self.yarn_get_mscale(self.rope_scale, float(self.mscale)) /
            self.yarn_get_mscale(self.rope_scale, float(self.mscale_all_dim)))

        self.original_max_position_embeddings = config.rope_scaling.get(
            'original_max_position_embeddings', 16384)
        self.max_position_embeddings = config.max_position_embeddings
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        self.low, self.high = self._yarn_find_correction_range(self.beta_fast,
                                                               self.beta_slow,
                                                               head_dim,
                                                               self.rope_theta,
                                                               self.original_max_position_embeddings)
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, 'partial_rotary_factor') else 1.0
        self.rotary_dim = int(head_dim * partial_rotary_factor)

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

    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 indptr: torch.Tensor,
                 offsets: torch.Tensor):
        flood_cuda.apply_yarn_rope_inplace(
            q, k, indptr, offsets, self.rotary_dim, False, self.rope_scale, self.rope_theta,
            self.low, self.high, self.magnitude_scale)
        return q, k

class DeepseekYarnRope(torch.nn.Module):
    def __init__(self, config):
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

        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        query_rot = query_rot * cos + self.rotate_gptj(query_rot) * sin
        key_rot = key_rot * cos + self.rotate_gptj(key_rot) * sin

        if self.rotary_dim < self.head_dim:
            q = torch.cat((query_rot, query_pass), dim=-1)
            k = torch.cat((key_rot, key_pass), dim=-1)
        else:
            q = query_rot
            k = key_rot
        return q, k.squeeze()


class Llama31Rope(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rope_scale = config.rope_scaling['factor']
        self.low_freq_factor = config.rope_scaling.get('low_freq_factor', 1.0)
        self.high_freq_factor = config.rope_scaling.get('high_freq_factor', 4.0)
        self.original_max_position_embeddings = config.rope_scaling.get(
            'original_max_position_embeddings', 8192)

    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 indptr: torch.Tensor,
                 offsets: torch.Tensor):
        flood_cuda.apply_llama31_rope_inplace(
            q, k, indptr, offsets, False, self.rope_scale, self.rope_theta,
            self.low_freq_factor, self.high_freq_factor,
            self.original_max_position_embeddings)

        return q, k
