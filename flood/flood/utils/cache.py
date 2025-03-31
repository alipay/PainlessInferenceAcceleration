# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import List, Tuple

import torch
from transformers.cache_utils import Cache

from flood.ops import update_cache, quant_and_update_cache
from flood.utils.batch import Batch


class SegmentCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, max_token: int, num_layers: int = 32, 
                 num_key_value_heads: int = 8, head_dim: int = 128, dtype=None,
                 devices=()) -> None:
        super().__init__()
        self.max_token = max_token
        self.num_layers = num_layers
        self.head_dim = head_dim

        self.num_key_value_heads = num_key_value_heads
        self.dtype = dtype

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_token, self.num_key_value_heads, self.head_dim)
        for i in range(num_layers):
            kc = torch.zeros(cache_shape, dtype=self.dtype,
                             device=devices[i]).share_memory_()
            vc = torch.zeros(cache_shape, dtype=self.dtype,
                             device=devices[i]).share_memory_()
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            torch._dynamo.mark_static_address(kc)
            torch._dynamo.mark_static_address(vc)
            self.key_cache.append(kc)
            self.value_cache.append(vc)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            batch_meta_info: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        indices = batch_meta_info.cache_indices
        update_cache(k_out, v_out,
                     key_states, value_states,
                     indices)

        return k_out, v_out

    def quant_to_fp8_and_update(
            self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            batch_meta_info: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        indices = batch_meta_info.cache_indices
        q_out = torch.empty(query_states.shape, dtype=torch.float8_e4m3fn,
                            device=query_states.device)
        quant_and_update_cache(q_out,
                               k_out,
                               v_out,
                               query_states,
                               key_states,
                               value_states,
                               indices)
        return q_out, k_out, v_out
