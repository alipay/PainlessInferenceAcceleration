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
import warnings
import pandas as pd
import os
import sys
import numpy as np
import pickle
import json
from rouge_score import rouge_scorer

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.autograd import Variable
from transformers.activations import gelu
import torch.nn.functional as F

from ipad.models.glm.tokenization_glm import GLMChineseTokenizer
from ipad.models.glm.modeling_glm import GLMForConditionalGeneration
from ipad.common.sparse_module import GptSparseAttn, GptSparseMLP,SparseLayerNorm,GptSparseDim
from ipad.common.distill_worker import DistillPipe, DistillWorker

torch.manual_seed(7)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GlmDistillWorker(DistillWorker):
    attn_name = 'attention'
    mlp_name = 'mlp'
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'
    def __init__(self,
                 sample_dir=None, logit_dir=None, emb_dir=None, emb_idx=1,
                 aux_sample_dir=None, aux_logit_dir=None, aux_emb_dir=None,
                 log_dir=None, log_steps=1, eval_steps=5, eval_count=100, error_count=0,
                 max_input_len=256, max_gen_len=64,
                 train_dtype=torch.bfloat16,
                 pred_dtype=torch.float16,
                 device=None):
        super(GlmDistillWorker, self).__init__(log_dir=log_dir)
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
        self.error_count = error_count
        self.max_input_len = max_input_len
        self.max_gen_len = max_gen_len
        self.train_dtype = train_dtype
        self.pred_dtype = pred_dtype
        self.device = device

    def _model_from_pretrained(self, model_dir, dtype=torch.bfloat16):
        model = GLMForConditionalGeneration.from_pretrained(model_dir,
                                                            cache_dir='/',
                                                            torch_dtype=dtype,
                                                            device_map="auto")
        return model

    def _tokenizer_from_pretrained(self, token_dir):
        tokenizer = GLMChineseTokenizer.from_pretrained(token_dir)
        tokenizer.padding_side='left'
        return tokenizer

    def _get_sparse_norm(self, layer=None, mask=None):
        return SparseLayerNorm(layer=layer, mask=mask)

    def _get_sparse_mlp(self, layer=None, layer_index=0):
        return GptSparseMLP(layer=layer, layer_index=layer_index)

    def _get_sparse_attn(self, layer=None, layer_index=0):
        return GptSparseAttn(layer=layer, layer_idx=layer_index)

    def _get_sparse_dim(self, layer=None, layer_index=0, mask=None):
        return GptSparseDim(layer=layer, layer_idx=layer_index, mask=mask)

    @property
    def n_layer(self):
        return len(self.model.glm.transformer.layers)

    def get_layers(self):
        return self.model.glm.transformer.layers

    def set_layers(self, layers):
        self.model.glm.transformer.layers = layers

    def get_final_norm(self):
        return self.model.glm.transformer.final_layernorm

    def set_final_norm(self, norm):
        self.model.glm.transformer.final_layernorm = norm

    def get_transformer(self):
        return self.model.glm.transformer

    def set_transformer(self, transformer):
        self.model.glm.transformer = transformer

    def get_head(self):
        return self.model.glm.lm_head

    def set_head(self, head):
        self.model.glm.lm_head = head

    def get_wte(self):
        return self.model.glm.word_embeddings

    def set_wte(self, wte):
        self.model.glm.word_embeddings = wte

    def get_wpe(self):
        return [self.model.glm.transformer.position_embeddings, self.model.glm.transformer.block_position_embeddings]

    def set_wpe(self, wpe):
        self.model.glm.transformer.position_embeddings = wpe[0]
        self.model.glm.transformer.block_position_embeddings = wpe[1]

    def forward(self, input_ids, position_ids=None, attention_mask=None, embeddings=None, index=1):

        input_embs = self.get_wte()(input_ids)
        if embeddings is not None:
            length = embeddings.size(1)
            input_embs[:, index:index + length] = embeddings

        outputs = self.get_transformer()(input_embs,
                                         position_ids=position_ids,
                                         attention_mask=attention_mask)
        return outputs[0]

    def save_conf(self, save_dir=None, prefix='GLMConfig'):
        ts = time.time()
        conf = self.model.config

        conf.num_hidden_layers = self.n_layer

        conf.hidden_size = self.get_head().weight.size(1)
        mlp = getattr(self.get_layers()[0], self.mlp_name)
        l1 = mlp.dense_h_to_4h
        conf.intermediate_size = l1.weight.size(0)

        attn = getattr(self.get_layers()[0], self.attn_name)
        l1 = attn.query_key_value
        conf.attention_size = l1.weight.size(0) // 3
        conf.tie_word_embeddings = False

        conf_str = str(conf)[len(prefix) + 1:]
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + '/config.json', 'w') as f:
            f.write(conf_str)

        self.log(f'save conf:{save_dir.split(",")[-1]} in {round(time.time() - ts, 3)}s')

    def tokenize(self, queries=None, answers=None, max_input_len=256, max_gen_len=128):
        if answers is not None:
            queries = [x if '[gMASK]' in x else x + '[gMASK]' for x in queries]
            inputs = self.tokenizer(queries,
                                    padding="max_length",
                                    max_length=max_input_len,
                                    return_tensors="pt",
                                    truncation=True)
            inputs = self.tokenizer.build_inputs_for_generation(inputs,
                                                                targets=answers,
                                                                max_gen_length=max_gen_len,
                                                                padding=True)
            attention_mask = inputs['attention_mask'].to(self.device)
            labels = inputs['labels'].to(self.device)
        else:
            queries = [x if '[gMASK]' in x else x + '[gMASK]' for x in queries]
            inputs = self.tokenizer(queries,
                                    return_tensors="pt",
                                    padding='max_length',
                                    max_length=max_input_len,
                                    truncation=True
                                    )
            inputs = self.tokenizer.build_inputs_for_generation(inputs,
                                                                max_gen_length=max_gen_len
                                                                )
            attention_mask = inputs['generation_attention_mask'].to(self.device)
            labels = None

        input_ids = inputs['input_ids'].to(self.device)
        position_ids = inputs['position_ids'].to(self.device)

        inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return inputs

    def chat(self, queries, embeddings=None, max_length=256, strip=True):
        inputs = self.tokenize(queries, max_input_len=self.max_input_len, max_gen_len=max_length)
        # print(f'{queries=} {inputs=}')
        input_ids = inputs['input_ids']
        input_length = input_ids.shape[1]
        attention_mask = inputs['attention_mask']
        if embeddings is not None:
            embeddings = torch.from_numpy(embeddings).to(attention_mask.device)
        if self.input_emb_mask is not None:
            embeddings = embeddings[:,:, self.input_emb_mask==1.0]
        outputs = self.model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      position_ids=inputs['position_ids'],
                                      inputs_embeds=embeddings,
                                      use_cache=True,
                                      max_length=max_length + input_length,
                                      repetition_penalty=1.0,
                                      do_sample=False,
                                      temperature=1.0,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      bos_token_id=self.tokenizer.eos_token_id,
                                      #  top_k=10,
                                      #  num_beams=1,
                                      #  top_p=0.9,
                                      # num_beams=3,
                                      # num_return_sequences=1,
                                      )
        io_ids = outputs.tolist()
        input_ids = [x[:input_length] for x in io_ids]
        output_ids = [x[input_length:] for x in io_ids]
        input_texts = []
        output_texts = []
        eos = 50005
        for i, o_ids in enumerate(output_ids):
            if eos in o_ids:
                o_ids = o_ids[:o_ids.index(eos) + 1]
            input_text = self.tokenizer.decode(input_ids[i])
            input_texts.append(input_text)
            output_text = self.tokenizer.decode(o_ids)
            if strip:
                output_text = output_text.replace("<|startofpiece|>", " ").replace("<|endofpiece|>", " ")
            output_texts.append(output_text)
        return input_texts, input_ids, attention_mask, output_ids, output_texts

    def calc_acc(self, count=500, batch_size=16, ds='test', max_log_count=0, filename=None, info=''):
        assert ds in ('aux', 'test')
        ts = time.time()
        queries = self.test_queries if ds == 'test' else self.aux_queries
        answers = self.test_answers if ds == 'test' else self.aux_answers

        checkpoint = self.get_transformer().gradient_checkpointing
        self.get_transformer().gradient_checkpointing = False
        rs = self.batch_chat(queries[:count],
                             ans=answers[:count],
                             batch_size=batch_size, 
                             max_length=self.max_gen_len,
                             max_log_count=max_log_count,
                             log_query=False,
                             log_truth=False,
                             log_answer=True,
                             log_space=False,
                             emit=True)
        self.get_transformer().gradient_checkpointing = checkpoint

        if filename is not None:
            jsons = []
            for i, q in enumerate(queries[:count]):
                r = rs[i]
                a = answers[i]
                jsons.append(json.dumps({'prompt':q, 'response': r, 'truth': a}))
            with open(filename, 'w') as f:
                f.write('\n'.join(jsons))
        
        te = time.time()
        n_p = 0
        n = 0
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for i, ans in enumerate(rs):
            pred = ans.replace('<|endofpiece|>', '').strip()
            true = answers[i].replace('<|endofpiece|>', '').strip()
            n += 1

            score = scorer.score(prediction=' '.join(pred), target=' '.join(true))["rougeL"].fmeasure
            n_p += score
        self.log(f'\nrouge:{n_p / max(n,1):.3f} sample:{n} elapse:{te - ts:.3f} {info}\n')
        acc = n_p / max(n, 1)
        return acc

