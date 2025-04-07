# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import List, Tuple

from huggingface_hub import cached_assets_path
import torch
from transformers.cache_utils import Cache

from flood.ops.cache import update_cache, update_fusion_cache, quant_and_update_cache
from flood.utils.batch import Batch


class SegmentCache(Cache):

    def __init__(self, 
                 max_token: int, 
                 num_layers: int = 32, 
                 dims: List[int] = [1024,1024], 
                 dtype=None,
                 devices=()) -> None:
        super().__init__()
        self.max_token = max_token
        self.num_layers = num_layers
        self.dims = dims  # key cache dim + value cache dim, 2048=2*8*128

        self.dtype = dtype

        self.caches: List[torch.Tensor] = []
        cache_shape = (max_token, sum(self.dims))
        for i in range(num_layers):
            cache = torch.zeros(cache_shape, dtype=self.dtype,
                             device=devices[i]).share_memory_()
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            torch._dynamo.mark_static_address(cache)
            self.caches.append(cache)

    def update_cache(
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
            batch_meta_info (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache = self.caches[layer_idx]
        head_dim = key_states.size(-1)
        key_cache, value_cache = cache.split(self.dims, dim=-1)

        indices = batch_meta_info.cache_indices
        update_cache(key_cache,
                     value_cache,
                     key_states, 
                     value_states,
                     indices)

        return key_cache.view(-1,self.dims[0]//head_dim, head_dim), value_cache.view(-1,self.dims[1]//head_dim, head_dim)

    def update_fusion_cache(
            self,
            key_value_states: torch.Tensor,
            layer_idx: int,
            batch_meta_info: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_value_states (`torch.Tensor`):
                The new key and value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            batch_meta_info (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache = self.caches[layer_idx]

        indices = batch_meta_info.cache_indices
        update_fusion_cache(cache,
                     key_value_states, 
                     indices)

        return cache

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
        cache = self.caches[layer_idx]
        key_cache, value_cache = cache.split(self.dims, dim=-1)
        head_dim = key_states.size(-1)

        indices = batch_meta_info.cache_indices
        q_out = torch.empty(query_states.shape, 
                            dtype=torch.float8_e4m3fn,
                            device=query_states.device)
        quant_and_update_cache(q_out,
                               key_cache,
                               value_cache,
                               query_states,
                               key_states,
                               value_states,
                               indices)
        return q_out, key_cache.view(-1,self.dims[0]//head_dim, head_dim), value_cache.view(-1,self.dims[1]//head_dim, head_dim)
