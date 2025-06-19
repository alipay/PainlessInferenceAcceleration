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
        self.rotaty_dim = int(head_dim * partial_rotary_factor)


    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 indptr: torch.Tensor,
                 offsets: torch.Tensor):
        flood_cuda.apply_rope_inplace(q, k, indptr, offsets, self.rotaty_dim, False,
                                      self.rope_scale, self.rope_theta)
        return q, k


class YarnRope(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rope_scale = config.rope_scaling.get('factor', 4.0)
        self.beta_slow = config.rope_scaling.get('beta_slow', 1.0)
        self.beta_fast = config.rope_scaling.get('beta_fast', 32.0)
        self.head_dim = config.rope_scaling.get('beta_fast', 32.0)
        self.magnitude_scale = float(
            0.1 * math.log(config.rope_scaling.get('factor', 4.0)) + 1.0)
        self.original_max_position_embeddings = config.rope_scaling.get(
            'original_max_position_embeddings', 16384)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        self.low, self.high = self._yarn_find_correction_range(self.beta_fast,
                                                               self.beta_slow,
                                                               head_dim,
                                                               self.rope_theta,
                                                               self.original_max_position_embeddings)

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
            q, k, indptr, offsets, False, self.rope_scale, self.rope_theta,
            self.low, self.high, self.magnitude_scale)
        return q, k


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
