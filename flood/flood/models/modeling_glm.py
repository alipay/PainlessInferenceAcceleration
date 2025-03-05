# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
import time
from typing import List, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel

from flood.layers.attention import AutoAttention
from flood.layers.embedding import AutoEmbedding
from flood.layers.linear import AutoLinear
from flood.layers.rope import AutoRope
from flood.layers.sampler import Sampler
from flood.ops import RMSNorm, silu_and_mul
from flood.utils.batch import Batch
from .configuration_glm import GLMConfig


class GLMMLP(torch.nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = AutoLinear.from_pretrained(self.hidden_size,
                                                    self.intermediate_size,
                                                    bias=config.use_bias,
                                                    config=config,
                                                    name='gate_proj')
        self.up_proj = AutoLinear.from_pretrained(self.hidden_size,
                                                  self.intermediate_size,
                                                  bias=config.use_bias,
                                                  config=config,
                                                  name='up_proj')
        self.down_proj = AutoLinear.from_pretrained(self.intermediate_size,
                                                    self.hidden_size,
                                                    bias=config.use_bias,
                                                    config=config,
                                                    name='down_proj')

    def flood_patch_func(self, kwargs=None):
        if self.layer_idx == 0:
            print('patch MLP')

        self.gate_up_proj = self.gate_proj.merge([self.gate_proj, self.up_proj])
        self.down_proj.patch()

        self.gate_proj = None
        delattr(self, 'gate_proj')

        self.up_proj = None
        delattr(self, 'up_proj')

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        act = silu_and_mul(gate_up)
        return self.down_proj(act)


class GLMAttention(torch.nn.Module):
    def __init__(self, config: GLMConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim',
                                self.hidden_size // self.num_heads)
        self.intermediate_size = self.num_heads * self.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = float(config.rope_theta)
        self.softmax_scale = math.sqrt(1.0 / self.head_dim)

        self.query_key_value = AutoLinear.from_pretrained(self.hidden_size,
                                                          (
                                                                      self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
                                                          bias=config.use_qkv_bias,
                                                          config=config,
                                                          name='query_key_value')
        self.dense = AutoLinear.from_pretrained(self.intermediate_size,
                                                self.hidden_size,
                                                bias=config.use_bias,
                                                config=config,
                                                name='dense')

        self.rope = AutoRope.from_pretrained(config)

    def flood_patch_func(self, kwargs=None):
        cache_dtype = kwargs[
            'cache_dtype'] if kwargs is not None and 'cache_dtype' in kwargs else None
        interleave_value = kwargs.get('interleave_value',
                                      False) if kwargs is not None else False
        if interleave_value:
            permute = []
            for g in range(self.num_key_value_heads):
                offset = (
                                     self.num_heads + self.num_key_value_heads + g) * self.head_dim
                for i in range(self.head_dim // 16):
                    for j in range(8):
                        permute.append(offset + i * 16 + j)
                        permute.append(offset + i * 16 + j + 8)
            permute = torch.tensor(permute, dtype=torch.int32,
                                   device=self.query_key_value.weight.data.device)
            offset = (self.num_heads + self.num_key_value_heads) * self.head_dim
            if self.query_key_value.weight.data.dtype == torch.float8_e4m3fn:
                self.query_key_value.weight.data.view(torch.int8)[:,
                offset:] = self.query_key_value.weight.data.view(torch.int8)[:,
                           permute]
            else:
                self.query_key_value.weight.data[offset:] = \
                self.query_key_value.weight.data[permute]
            if self.query_key_value.bias is not None:
                self.query_key_value.bias.data[offset:] = \
                self.query_key_value.bias.data[permute]

        self.attention = AutoAttention.from_pretrained(cache_dtype,
                                                       layer_idx=self.layer_idx,
                                                       kernels=kwargs.get('kernels',['sa']),
                                                       softmax_scale=self.softmax_scale)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            **kwargs,
    ) -> torch.Tensor:

        q_len = hidden_states.size(0)
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(q_len, self.num_heads + 2 * self.num_key_value_heads,
                       self.head_dim)

        query_states, key_states, value_states = qkv.split([self.num_heads,
                                                            self.num_key_value_heads,
                                                            self.num_key_value_heads],
                                                           dim=-2)

        batch_meta_info = kwargs['batch_meta_info']

        self.rope(query_states, key_states, batch_meta_info.q_offsets,
                  batch_meta_info.pids)

        attn_output = self.attention(query_states, key_states, value_states,
                                     batch_meta_info, past_key_value)

        attn_output = attn_output.view(q_len,
                                       self.intermediate_size)  # moe may have different hidden_sizes
        attn_output = self.dense(attn_output)

        return attn_output


class GLMDecoderLayer(torch.nn.Module):
    def __init__(self, config: GLMConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = GLMAttention(config, layer_idx=layer_idx)

        self.mlp = GLMMLP(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            batch_meta_info: Optional[Batch] = None,
            cutoff: Optional[bool] = False
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_meta_info=batch_meta_info
        )

        hidden_states += residual

        if cutoff:
            indices = batch_meta_info.q_offsets[1:] - 1
            hidden_states = hidden_states[indices]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states += residual

        return hidden_states


class GLMModel(PreTrainedModel):
    config_class = GLMConfig
    base_model_prefix = "model"
    _no_split_modules = ["GLMDecoderLayer"]

    def __init__(self, config: GLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.word_embeddings = AutoEmbedding.from_pretrained(config,
                                                             config.vocab_size,
                                                             config.hidden_size,
                                                             padding_idx=self.padding_idx)
        self.transformer = torch.nn.Module()
        self.transformer.layers = torch.nn.ModuleList(
            [GLMDecoderLayer(config, layer_idx=layer_idx) for layer_idx in
             range(config.num_layers)]
        )
        self.transformer.final_layernorm = RMSNorm(config.hidden_size,
                                                   eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def flood_patch_func(self, kwargs=None):
        self.layers = self.transformer.layers
        self.norm = self.transformer.final_layernorm


class GLMForCausalLM(PreTrainedModel):
    config_class = GLMConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.glm = GLMModel(config)
        self.vocab_size = config.vocab_size
        # self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head = AutoLinear.from_pretrained(config.hidden_size,
                                                  config.vocab_size,
                                                  bias=False,
                                                  config=config,
                                                  name='lm_head')
        self.sampler = Sampler()

    def get_input_embeddings(self):
        return self.glm.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def flood_patch_func(self, kwargs=None):
        if hasattr(self.config, 'norm_head') and self.config.norm_head:
            norm = torch.norm(self.lm_head.weight.data, p=2, dim=0,
                              keepdim=True) + 1e-7
            self.lm_head.weight.data /= norm
        if hasattr(self.lm_head, 'patch'):
            self.lm_head.patch()
        self.model = self.glm

    @torch.inference_mode()
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            batch_meta_info: Batch = None,
            device_list: List = None,
            sync_layers: List = None,
            streams: List = None
    ) -> List:

        n_devices = len(device_list)
        n_layers = sum([len(x) for x in device_list])
        ts = time.time()
        for i, indices in enumerate(device_list):
            stream = streams[i]
            with torch.cuda.stream(stream):
                if i == 0:
                    batch_meta_info.to(torch.device(0), non_blocking=True)
                    hidden_states = self.model.word_embeddings(
                        batch_meta_info.input_ids)
                    embeddings = batch_meta_info.embeddings
                    if embeddings is not None:
                        emb_idx_list = batch_meta_info.emb_idx_list
                        for ie, emb_idx in enumerate(emb_idx_list):
                            if emb_idx is None:
                                continue
                            ss, se, ds, de = emb_idx
                            hidden_states[ds:de] = embeddings[ie][ss:se]
                sync_layers[i]()

                for j in indices:
                    cutoff = True if j == n_layers - 1 and batch_meta_info.mode in (
                    0, 10) else False
                    hidden_states = self.model.layers[j](
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        batch_meta_info=batch_meta_info,
                        cutoff=cutoff
                    )

                if i < n_devices - 1:
                    device = torch.device(i + 1)
                    hidden_states = hidden_states.to(device, non_blocking=True)
                    batch_meta_info.to(device, non_blocking=True)
                else:
                    hidden_states = self.model.norm(hidden_states)
                    logits = self.lm_head(hidden_states)
                    next_token_id_list = self.sampler(logits,
                                                      batch_meta_info=batch_meta_info)

                stream.synchronize()

        sync_layers[-1]()

        return next_token_id_list
