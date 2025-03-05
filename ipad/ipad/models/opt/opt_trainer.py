# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


from __future__ import print_function

import time
from operator import itemgetter
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import copy
import types
import random
import math
import warnings
import os
import sys
import pickle
import json
import shutil
import pandas as pd

import numpy as np
from rouge_score import rouge_scorer

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.autograd import Variable
from transformers import DataCollatorWithPadding, BatchEncoding
import torch.nn.functional as F

from transformers import GPT2TokenizerFast
from ipad.models.opt.modeling_opt import OPTForCausalLM
from ipad.common.distill_worker import DistillWorker, DistillPipe
from ipad.common.sparse_module import GptSparseMLP, GptSparseAttn, GptSparseDim, SparseLayerNorm

torch.manual_seed(7)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class OptSparseMLP(GptSparseMLP):
    up_name = 'fc1'
    down_name = 'fc2'

    def set_act(self):
        self.act = F.relu


class OptSparseAttn(GptSparseAttn):
    q_name = 'q_proj'
    k_name = 'k_proj'
    v_name = 'v_proj'
    o_name = 'out_proj'
    qkv_name = None

    def set_dim(self, layer):
        self.hidden_size = layer.embed_dim
        self.head_dim = layer.head_dim
        self.num_heads = layer.num_heads
        self.n_repeat = 1


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        return super().forward(hidden_states, past_key_value=past_key_value,attention_mask=attention_mask, use_cache=True)


class OptSparseDim(GptSparseDim):
    ln1_name = 'self_attn_layer_norm'
    ln2_name = 'final_layer_norm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'
    ln_type = 'ln'

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ):

        ln1 = self.get(self.ln1_name)
        ln2 = self.get(self.ln2_name)
        attn = self.get(self.attn_name)
        mlp = self.get(self.mlp_name)

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        hidden_states = ln1(hidden_states)

        if self.training:
            hidden_states = hidden_states * self.mask

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        hidden_states = ln2(hidden_states)

        # hidden_states = self.fc1(hidden_states)
        # hidden_states = self.activation_fn(hidden_states)
        # hidden_states = self.fc2(hidden_states)

        if self.training:
            hidden_states = hidden_states * self.mask

        hidden_states = mlp(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OptDistillWorker(DistillWorker):
    ln1_name = 'self_attn_layer_norm'
    ln2_name = 'final_layer_norm'
    attn_name = 'self_attn'
    mlp_name = 'mlp'

    def __init__(self,
                 sample_dir=None, logit_dir=None, emb_dir=None, emb_idx=1,
                 aux_sample_dir=None, aux_logit_dir=None, aux_emb_dir=None,
                 log_dir=None, log_steps=1, eval_steps=5, eval_count=100,
                 max_input_len=256, max_gen_len=64,
                 train_dtype=torch.bfloat16,
                 pred_dtype=torch.float16,
                 device=None):
        super(OptDistillWorker, self).__init__(log_dir=log_dir)
        self.sample_dir = sample_dir
        self.logit_dir = logit_dir
        self.emb_dir = emb_dir
        self.emb_idx = emb_idx
        self.aux_sample_dir = aux_sample_dir
        self.aux_logit_dir = aux_logit_dir
        self.aux_emb_dir = aux_emb_dir
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.eval_count = eval_count
        self.max_input_len = max_input_len
        self.max_gen_len = max_gen_len
        self.train_dtype = train_dtype
        self.pred_dtype = pred_dtype
        self.device = device

    def _model_from_pretrained(self, model_dir, dtype=torch.bfloat16):
        model = OPTForCausalLM.from_pretrained(model_dir,
                                               cache_dir='/',
                                               torch_dtype=dtype,
                                               device_map="auto")
        return model

    def _tokenizer_from_pretrained(self, token_dir):
        tokenizer = GPT2TokenizerFast.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_sparse_norm(self, layer=None, mask=None):
        return SparseLayerNorm(layer=layer, mask=mask)

    def _get_sparse_mlp(self, layer=None, layer_index=0):
        return OptSparseMLP(layer=layer, layer_index=layer_index)

    def _get_sparse_attn(self, layer=None, layer_index=0):
        return OptSparseAttn(layer=layer, layer_idx=layer_index)

    def _get_sparse_dim(self, layer=None, layer_index=0, mask=None):
        return OptSparseDim(layer=layer, layer_idx=layer_index, mask=mask)

    def get_layers(self):
        return self.model.model.decoder.layers

    def set_layers(self, layers):
        self.model.model.decoder.layers = layers

    def get_final_norm(self):
        return self.model.model.decoder.final_layer_norm

    def set_final_norm(self, norm):
        pass

    def get_transformer(self):
        return self.model.model.decoder

    def set_transformer(self, transformer):
        self.model.model.decoder = transformer

    def get_head(self):
        return self.model.lm_head

    def set_head(self, head):
        self.model.lm_head = head

    def get_wte(self):
        return self.model.model.decoder.embed_tokens

    def set_wte(self, wte):
        self.model.model.decoder.embed_tokens = wte

    def get_wpe(self):
        return self.model.model.decoder.embed_positions

    def set_wpe(self, wpe):
        self.model.model.decoder.embed_positions = wpe
    @property
    def n_layer(self):
        return len(self.model.model.decoder.layers)

    def forward(self, input_ids, position_ids=None, attention_mask=None, embeddings=None):

        outputs = self.get_transformer().forward(input_ids,
                                                 attention_mask=attention_mask)
        return outputs[0]

    def est_params(self, head=False, tied=True):
        n_params = 0
        depth = self.n_layer
        dim = self.dim - self.mask_counts.get('dim', 0)
        attn_dim = self.attn_dim - self.mask_counts.get('attn', 0)
        mlp_dim = self.mlp_dim - self.mask_counts.get('mlp', 0)
        if head:
            n_params += self.n_voc * dim * (1 if tied else 2)
        n_params += (2 * dim * mlp_dim + mlp_dim + dim) * depth
        n_params += ( 4 * dim * attn_dim + 3 * attn_dim + dim) * depth
        n_params += (depth * 2 + 1) * dim  # layernorm
        return n_params

    def save_conf(self, save_dir=None, prefix='OptConfig'):
        ts = time.time()
        conf = self.model.config

        conf.num_hidden_layers = self.n_layer

        conf.hidden_size = self.get_head().weight.size(1)
        mlp = getattr(self.get_layers()[0], self.mlp_name)
        l1 = mlp.fc1
        conf.intermediate_size = l1.weight.size(0)

        attn = getattr(self.get_layers()[0], self.attn_name)
        l1 = attn.q_proj
        conf.attention_size = l1.weight.size(0)  # TODO
        conf.tie_word_embeddings = False

        conf_str = str(conf)[len(prefix) + 1:]
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + '/config.json', 'w') as f:
            f.write(conf_str)

        self.log(f'save conf:{save_dir.split(",")[-1]} in {round(time.time() - ts, 3)}s')

    def calc_acc(self, count=100, batch_size=1, ds='test', max_log_count=0, filename=None, info=''):
        assert ds in ('test', 'train', 'aux')
        count = 1000000 if count is None else count

        ts = time.time()
        embeddings = None
        queries = None
        answers = None
        if ds == 'test':
            queries = self.test_queries
            answers = self.test_answers
            embeddings = self.test_embeddings
        elif ds == 'train':
            queries = self.queries
            answers = self.answers
            embeddings = self.embeddings
        elif ds == 'aux':
            queries = self.aux_queries
            answers = self.aux_answers
            embeddings = self.aux_embeddings

        if embeddings is not None:
            embeddings = embeddings[:count]

        checkpoint = self.get_transformer().gradient_checkpointing
        self.get_transformer().gradient_checkpointing = False
        rs = self.batch_chat(queries[:count],
                             embeddings=embeddings,
                             batch_size=batch_size,
                             max_length=self.max_gen_len,
                             max_log_count=max_log_count,
                             log_query=False,
                             log_truth=False,
                             log_answer=True,
                             log_space=False,
                             emit=True)
        te = time.time()
        
        self.get_transformer().gradient_checkpointing = checkpoint

        if filename is not None:
            jsons = []
            for i, q in enumerate(queries[:count]):
                r = rs[i]
                a = answers[i]
                jsons.append(json.dumps({'prompt':q, 'response': r, 'truth': a}))
            with open(filename, 'w') as f:
                f.write('\n'.join(jsons))

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        te = time.time()
        n_p = 0
        n = 0
        n_ps = [0] * ((len(rs) - 1) // 100 + 1)
        score_details = [[],[]]
        for i, ans in enumerate(rs):
            pred = ans.replace('<s>', '').replace('</s>', '').strip()
            true = answers[i].replace('<s>', '').replace('</s>', '').strip()
            n += 1
            score = scorer.score(prediction=pred, target=true)["rougeL"].fmeasure
            n_p += score
            n_ps[i // 100] += score
            if 'paired with an input' in queries[i]:
                score_details[0].append(score)
            else:
                score_details[1].append(score)
        acc = n_p / n
        details = [round(x / 100, 3) for x in n_ps]
        score_details = [round(sum(x)/max(len(x), 3),1) for x in score_details]
        size = self.est_params()/1e9
        self.log(f'\nrouge:{acc:.3f} sample:{n} details:{details}/{score_details} size:{size:.3f} elapse:{te - ts:.3f} {info}\n')
        return acc
