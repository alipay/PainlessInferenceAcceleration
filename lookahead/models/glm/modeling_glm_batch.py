# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import time
import copy
import json
import pickle

import warnings
from functools import reduce
from collections import defaultdict
from itertools import accumulate
import inspect
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import sys


import numpy as np
import torch
import torch.utils.checkpoint
from torch.nn.utils import skip_init
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# from transformers.activations import gelu
from torch.nn.functional import gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
# from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.modeling_utils import SequenceSummary
from common.pretrained_model_batch import LookaheadPreTrainedModel, PrefetchCache

from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GreedySearchOutput, ModelOutput, \
    validate_stopping_criteria, GreedySearchDecoderOnlyOutput


sys.path.append('/ossfs/workspace/lookahead/models')
from antglm.configuration_glm import GLMConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "glm"
_CONFIG_FOR_DOC = "GLMConfig"



class SelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.attention_size = config.attention_size if hasattr(config, 'attention_size') else config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.attention_size // self.num_heads
        self.norm_coef = self.head_dim ** (-0.5)
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.attention_size:
            raise ValueError(
                f"`attention_size` must be divisible by num_heads (got `attention_size`: {self.attention_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.attention_scale
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.layer_idx = layer_idx

        # self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.query_key_value = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)

        # self.attn_dropout = nn.Dropout(config.attention_dropout_prob)
        # self.resid_dropout = nn.Dropout(config.output_dropout_prob)

        self.norming = True
        self.normed = False

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):

        # attns = {}
        # attn_weights = torch.matmul(self.norm_coef*query, key.transpose(-1, -2))
        # attn_weights = attn_weights.masked_fill_(attention_mask==0.0, -65504.0)

        # print(attention_mask.shape, query.shape, key.shape)
        # coef is divide by weight and bias after loading
        if query.size(0) == 1:
            if self.normed:
                attn_weights = torch.baddbmm(attention_mask.squeeze(0), query.squeeze(0),
                                             key.squeeze(0).transpose(-1, -2))
            else:
                attn_weights = torch.baddbmm(attention_mask.squeeze(0), self.norm_coef * query.squeeze(0),
                                             key.squeeze(0).transpose(-1, -2))
        else:
            if self.normed:
                attn_weights = torch.matmul(key, query.transpose(-1, -2)).transpose(-1, -2)
            else:
                attn_weights = torch.matmul(key, self.norm_coef * query.transpose(-1, -2)).transpose(-1, -2)
            attn_weights = attn_weights.add_(attention_mask)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _fast_attn(self, query, key, value, attention_mask=None, head_mask=None):
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            return F.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            prefetch_kwargs: Optional[Dict] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        if self.norming and not self.normed:
            self.query_key_value.weight.requires_grad = False
            self.query_key_value.weight.data[:self.embed_dim] *= self.norm_coef
            self.query_key_value.bias.requires_grad = False
            self.query_key_value.bias.data[:self.embed_dim] *= self.norm_coef
            self.normed = True
            if self.layer_idx == 0:
                print('weight is normed! MUST NOT patch lora parameters afterwards!')

        # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        # query = self._split_heads(query, self.num_heads, self.head_dim)
        # key = self._split_heads(key, self.num_heads, self.head_dim)
        # value = self._split_heads(value, self.num_heads, self.head_dim)
        mat = self.query_key_value(hidden_states)
        new_shape = mat.size()[:-1] + (3, self.num_heads, self.head_dim)
        query, key, value = mat.view(new_shape).permute(0, 3, 2, 1, 4).unbind(2)

        if prefetch_kwargs is not None \
                and prefetch_kwargs.get('use_prefetch', False) \
                and LookaheadPreTrainedModel._batch_prefetch:
            if layer_past is None:
                attn_output, attn_weights = self._attn(query, key, value,
                                                       attention_mask=attention_mask)
            else:
                prefetch_cursors = prefetch_kwargs['prefetch_cursors']
                past_key, past_value = layer_past
                bs, l, _ = hidden_states.size()
                max_len = max(prefetch_cursors) + l
                cs = list(set(prefetch_cursors))
                if len(cs) == 1:
                    c = cs[0]
                    past_key[:, :, c: c + l] = key
                    past_value[:, :, c: c + l] = value
                else:
                    for i, cursor in enumerate(prefetch_cursors):
                        past_key[i, :, cursor: cursor + l] = key[i]
                        past_value[i, :, cursor: cursor + l] = value[i]
                key = past_key
                value = past_value
                attn_output, attn_weights = self._attn(query, key[:, :, :max_len], value[:, :, :max_len],
                                                       attention_mask=attention_mask)
        else:
            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.dense(attn_output)
        # attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GLMMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = intermediate_size
        self.dense_h_to_4h = nn.Linear(self.embed_dim, self.intermediate_size)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.embed_dim)
        self.act = gelu

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class VocabEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, config):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim))
        # And initialize.
        init.xavier_normal_(self.weight)

    def forward(self, input_):
        # Get the embeddings.
        output = F.embedding(input_, self.weight,
                             self.padding_idx, self.max_norm,
                             self.norm_type, self.scale_grad_by_freq,
                             self.sparse)
        return output


class GLMBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.bottleneck_size if hasattr(config, 'bottleneck_size') else 4 * hidden_size

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = GLMMLP(inner_dim, config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            prefetch_kwargs: Optional[Dict] = None
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.attention(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefetch_kwargs=prefetch_kwargs
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GLMStack(torch.nn.Module):
    _keys_to_ignore_on_load_missing = []

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_sequence_length + 1, self.embed_dim)
        self.block_position_embeddings = nn.Embedding(config.max_sequence_length + 1, self.embed_dim)
        self.layers = nn.ModuleList([GLMBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(self.embed_dim)

        # Model parallel
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = True,
            return_dict: Optional[bool] = None,
            prefetch_kwargs: Optional[Dict] = None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        attention_mask = -10000 * (1.0 - attention_mask.to(hidden_states.dtype))

        position_item_ids, position_block_ids = position_ids[:, :, :hidden_states.size(1)].unbind(1)
        position_embeds = self.position_embeddings(position_item_ids)
        position_block_embeds = self.block_position_embeddings(position_block_ids)
        hidden_states = hidden_states + position_embeds + position_block_embeds

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, False)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    prefetch_kwargs=prefetch_kwargs
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            # if output_attentions:
            #     all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            #     if self.config.add_cross_attention:
            #         all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.final_layernorm(hidden_states)

        # Add last hidden state
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )

class GLMPreTrainedModel(LookaheadPreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GLMConfig
    base_model_prefix = "glm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = []

    # def _init_weights(self, module):
    #     """ Initialize the weights """
    #     if isinstance(module, torch.nn.Linear):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, torch.nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, torch.nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value


class GLMModel(GLMPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    # base_model_prefix = "glm"
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.word_embeddings = VocabEmbedding(config)

        # Transformer
        self.transformer = GLMStack(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prefetch_kwargs: Optional[Dict] = None,
            inputs_embeds_position: Optional[int] = 1
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        words_embeddings = self.word_embeddings(input_ids)
        if inputs_embeds is not None and past_key_values is None:
            length = inputs_embeds.size(1)
            words_embeddings[:, inputs_embeds_position: inputs_embeds_position+length, :] = inputs_embeds

        transformer_outputs = self.transformer(
            words_embeddings,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            prefetch_kwargs=prefetch_kwargs
        )
        hidden_states = transformer_outputs[0]

        # 第一次前向只需要取最后一个hidden_states
        if past_key_values is None:
            hidden_states = hidden_states[:, -1:, :]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    # def get_output_embeddings(self):
    #     return self.lm_head

    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings

    # def get_input_embeddings(self):
    #     return self.word_embeddings

    # def set_input_embeddings(self, new_embeddings):
    #     self.word_embeddings = new_embeddings


class GLMForConditionalGeneration(GLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['glm.lm_head.weight']
    _no_split_modules = []
    # base_model_prefix = "glm"

    def __init__(self, config):
        super().__init__(config)
        self.glm = GLMModel(config)
        self.post_init()
        self.prefetch_cache = PrefetchCache()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prefetch_kwargs: Optional[Dict] = None,
            inputs_embeds_position: Optional[int] = 1
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        return self.glm(input_ids=input_ids,
                         position_ids=position_ids,
                         attention_mask=attention_mask,
                         inputs_embeds_position=inputs_embeds_position,
                         use_cache=use_cache,
                         past_key_values=past_key_values,
                         inputs_embeds=inputs_embeds,
                         labels=labels,
                         output_attentions=output_attentions,
                         output_hidden_states=output_hidden_states,
                         return_dict=return_dict,
                         prefetch_kwargs=prefetch_kwargs
                         )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask_cache = kwargs.get("attention_mask", None)
        position_ids_cache = kwargs.get("position_ids", None)
        model_inputs = {}

        if past_key_values is None:
            input_id_slice = input_ids
            input_length = input_ids.size(1)
            attention_mask = attention_mask_cache[:, :, :input_length, :input_length]
            position_ids = position_ids_cache[:, :, :input_length]
        else:
            prefix_length = input_ids.size(-1) - 1
            input_id_slice = input_ids[:,-1:]
            fresh_length = 1
            ppl = prefix_length + fresh_length
            attention_mask = attention_mask_cache[:, :, prefix_length:ppl, :ppl]
            position_ids = position_ids_cache[:, :, prefix_length:prefix_length + 1]

        model_inputs.update(
            {
                "input_ids": input_id_slice,
                "inputs_embeds": inputs_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        )
        return model_inputs

    def get_output_embeddings(self):
        return self.glm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.glm.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.glm.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.glm.word_embeddings = new_embeddings

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
            prefetched: bool = False
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        return model_kwargs

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args and key not in (
                    'use_prefetch', 'prefetch_size', 'prefetch_length', 'debug_prefetch', 'prefetch_kwargs',
                    'inputs_embeds_position'):
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            return past
        reordered_decoder_past = []
        for key, value in past:
            # get the correct batch idx from layer past batch dim
            reordered_decoder_past.append([
                key.index_select(0, beam_idx.to(key.device)),
                value.index_select(0, beam_idx.to(value.device))])
        return reordered_decoder_past


