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
import json
import shutil

import numpy as np
from rouge_score import rouge_scorer

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from transformers import BatchEncoding, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from .sparse_module import SparseMLP,SparseAttn,SparseNorm,SparseDim,SparseBlock


class DistillWorker():
    mlp_name = 'mlp'
    attn_name = 'self_attn'
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'

    def __init__(self,
                 sample_dir=None, logit_dir=None, emb_dir=None, emb_idx=1,
                 aux_sample_dir=None, aux_logit_dir=None, aux_emb_dir=None,
                 log_dir=None, log_steps=1, 
                 eval_steps=5, eval_count=100, error_count=0,
                 max_input_len=256, max_gen_len=64,
                 train_dtype=torch.bfloat16,
                 pred_dtype=torch.float16,
                 device=torch.device('cuda:0')):
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

        self.n_voc = None
        self.dim = None
        self.mlp_dim = None
        self.attn_dim = None
        self.num_layer = None

        self.layer_indices = None
        self.tokenizer = None
        self.model = None
        self.input_emb_mask = None

        self.queries = None
        self.answers = None
        self.embeddings = None
        self.logit_cache = None
        self.avg_logit_cache = None
        self.sample_indices = None

        self.test_queries = None
        self.test_answers = None
        self.test_embeddings = None
        self.test_logit_cache = None
        self.test_avg_logit_cache = None
        self.test_sample_indices = None

        if self.log_dir is not None:
            self.logger = open(self.log_dir + '_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.log', 'w')
        else:
            self.logger = None

        self.aux_queries = []
        self.aux_answers = []
        self.aux_embeddings = None
        self.aux_logit_cache = []
        self.aux_avg_logit_cache = []
        self.aux_sample_indices = []
        self.mask_counts = {}

        self.train_kernels = []

        self.optimizer = None
        self.trainable_params = None

    def load_model(self, model_dir, token_dir=None, mask_dir=None, distill=True, advantage=False, dtype=torch.float16):
        ts = time.time()
        token_dir = token_dir or model_dir
        tokenizer = self._tokenizer_from_pretrained(token_dir)
        self.log(f'load tokenizer in {round(time.time() - ts, 3)}s')
        self.tokenizer = tokenizer

        model = self._model_from_pretrained(model_dir, dtype=dtype)
        self.log(f'load model in {round(time.time() - ts, 3)}s')
        self.model = model.to(dtype).to(self.device)

        if distill:
            self.init_head_params()

            if advantage:
                self.model.advantage = nn.Linear(self.get_head().weight.size(1), 1).to(self.device)

            if mask_dir:
                self.input_emb_mask = self.load_mask(mask_dir)

        self.n_voc, self.dim = self.get_wte().weight.shape
        self.mlp_dim = self.model.config.intermediate_size if hasattr(self.model.config, 'intermediate_size') else 4*self.dim
        self.attn_dim = self.dim
        self.num_layer = self.n_layer
        print(f'{self.dim=} {self.mlp_dim=} {self.attn_dim=}')

    def init_head_params(self):
        self.model.fixed_head_weight = Parameter(self.get_head().weight.data.clone().detach(), requires_grad=False)
        if hasattr(self.model.config, 'tie_word_embeddings') and self.model.config.tie_word_embeddings:
            self.log('tie_word_embeddings=True, copy head weight')
            self.get_head().weight.data = copy.deepcopy(self.get_head().weight.detach())
        else:
            self.log('tie_word_embeddings=False, reuse head weight')

    def wait(self, use_wait, sleep=360):
        if use_wait:
            while True:
                if not os.path.exists(self.logit_dir):
                    for i in range(60):
                        torch.matmul(torch.rand(4096, 4096).to(self.device), torch.rand(4096, 4096).to(self.device))
                        time.sleep(0.1)
                else:
                    # wait for finishing writing
                    time.sleep(sleep)
                    break

    def _model_from_pretrained(self, model_dir, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     cache_dir='/',
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto")
        return model

    def _tokenizer_from_pretrained(self, token_dir):
        tokenizer = AutoTokenizer.from_pretrained(token_dir)
        return tokenizer

    def load_ckpt(self, ckpt_dir=None):
        dtype = self.train_dtype
        ts = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = self._model_from_pretrained(ckpt_dir, dtype=dtype)
        self.log(f'load model:{ckpt_dir} in {round(time.time() - ts, 3)}s')
        self.model = model.to(dtype).to(self.device)

        indices_list = [[], [], []]
        sd = torch.load(ckpt_dir + '/pytorch_model.bin', map_location=torch.device('cpu'))
        n_layer = len(self.get_layers())
        mlp_mask_counts = []
        attn_mask_counts = []
        dim_mask_count = 0
        for k, v in sd.items():
            if not k.endswith('mask') and k not in ('fixed_head_weight', 'advantage.weight', 'advantage.bias'):
                continue
            v = Parameter(v.to(dtype).to(self.device), requires_grad=False)
            if k == 'fixed_head_weight':
                self.model.fixed_head_weight = v.to(dtype).to(self.device)
            elif k.endswith(f'.{self.mlp_name}.mask'):
                idx = int(k.split('.')[-3])
                kernels = self.replace_kernels(mode='mlp', layer_indices=[idx])
                kernels[0].mask = v
                indices_list[0].append(idx)
                mlp_mask_counts.append((1-v).long().sum().item())
            elif k.endswith(f'.{self.attn_name}.mask'):
                idx = int(k.split('.')[-3])
                kernels = self.replace_kernels(mode='attn', layer_indices=[idx])
                kernels[0].mask = v
                indices_list[1].append(idx)
                attn_mask_counts.append((1-v[0]).long().sum().item())
            elif k == 'input_emb_mask':
                self.input_emb_mask = v
            elif k == 'mask':
                kernels = self.replace_kernels(mode='dim', layer_indices=range(n_layer), mask=v)
                self.model.mask = v
                self.set_final_norm(self._get_sparse_norm(layer=self.get_final_norm(), mask=v))
                indices_list[2].extend(range(n_layer))
                self.mask_counts['dim'] = (1-v).long().sum().item()
            elif k == 'advantage.weight':
                weight = Parameter(v.to(dtype).to(self.device), requires_grad=True)
                bias = Parameter(sd['advantage.bias'].to(dtype).to(self.device), requires_grad=True)
                self.model.advantage = nn.Linear(weight.size(1), 1)
                self.model.advantage.weight = weight
                self.model.advantage.bias = bias

        if len(mlp_mask_counts) > 0:
            self.mask_counts['mlp'] = sum(mlp_mask_counts)/len(mlp_mask_counts)
        if len(attn_mask_counts) > 0:
            self.mask_counts['attn'] = sum(attn_mask_counts)/len(attn_mask_counts)
        del sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log(f"mlp.mask:{indices_list[0]} attn.mask:{indices_list[1]} dim.mask:{indices_list[2]}")
        self.log(f'load mask in {round(time.time() - ts, 3)}s')

    def _get_sparse_norm(self, layer=None, mask=None):
        raise NotImplementedError('_get_sparse_norm')

    def _get_sparse_mlp(self, layer=None, layer_index=0):
        raise NotImplementedError('_get_sparse_mlp')

    def _get_sparse_attn(self, layer=None, layer_index=0):
        raise NotImplementedError('_get_sparse_attn')

    def _get_sparse_dim(self, layer=None, layer_index=0, mask=None):
        raise NotImplementedError('_get_sparse_dim')

    def load_mask(self, save_dir):
        mask = np.load(save_dir)['mask']
        return torch.from_numpy(mask).long().to(self.device) == 1

    def train_layer(self,
                    bs=8,
                    mbs=8,
                    pbs=8,
                    zero_counts=None,
                    max_input_len=None,
                    max_gen_len=None,
                    steps=None,
                    layer_indices=None,
                    lr=0.01,
                    lrs='cos',
                    optim='SGD',
                    check_state=None,
                    info='0',
                    mode='mlp',
                    loss_coefs=None,
                    ds='sft'):
        assert mode in ('mlp', 'act', 'attn', 'dim', 'depth', 'wte')
        assert lrs in ('cos', 'fix')
        assert optim in ('SGD', 'Adam', 'AdamW')
        assert ds in ('sft', 'aux')
        assert bs >= mbs and bs // mbs * mbs == bs
        steps = len(zero_counts) if zero_counts is not None else steps
        self.log(f'\n{ds=} {mode=} {bs=} {mbs=} {steps=} {lr=} {lrs=} {optim=} {loss_coefs=}')
        self.model.train()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=False)
        if layer_indices is None:
            layer_indices = list(range(len(self.get_layers())))
        self.layer_indices = layer_indices
        if mode == 'mlp':
            kernels = self.replace_kernels(mode='mlp', layer_indices=layer_indices)
        elif mode == 'attn':
            kernels = self.replace_kernels(mode='attn', layer_indices=layer_indices)
        elif mode == 'dim':
            """
            mask is shared with multiple steps, so second training starts with sparse mask
            """
            if not hasattr(self.model, 'mask'):
                dim = self.get_wte().weight.size(1)
                self.model.mask = Parameter(torch.ones((dim,), dtype=self.train_dtype).to(self.device),
                                            requires_grad=False)
            kernels = self.replace_kernels(mode='dim', layer_indices=layer_indices, mask=self.model.mask)
            if self.get_final_norm() is not None:
                self.set_final_norm(self._get_sparse_norm(layer=self.get_final_norm(), mask=self.model.mask))
        elif mode == 'depth':
            train_index = layer_indices[-1]
            assert train_index < self.n_layer
            if train_index < self.n_layer - 1:
                self.clip_layer(train_index+1)
            kernels = self.replace_kernels(mode='depth', layer_indices=[train_index])

            head = self.get_head()
            kernels.append(head)
            use_ln = True
            use_ln = use_ln and self.get_final_norm() is not None
            if use_ln:
                kernels.append(self.get_final_norm())
            for idx in layer_indices:
                kernels.append(self.get_layers()[idx])

        self.train_kernels = kernels
        trainable_params = self.get_params(kernels)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in trainable_params:
            p.requires_grad = True
        if optim == 'SGD':
            self.optimizer = SGD(trainable_params, lr=lr * mbs / bs)
        elif optim == 'Adam':
            self.optimizer = Adam(trainable_params, lr=lr * mbs / bs, betas=(0.9, 0.95))
        elif optim == 'AdamW':
            self.optimizer = AdamW(trainable_params, lr=lr * mbs / bs, betas=(0.9, 0.95), weight_decay=0.1)
        if lrs == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        51200 // bs,
                                                                        eta_min=0,
                                                                        last_epoch=-1,
                                                                        verbose=False)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1024, gamma=1.0)
        self.trainable_params = trainable_params
        self.get_transformer().gradient_checkpointing = False
        ret = self.fit(bs=bs,
                       mbs=mbs,
                       pbs=pbs,
                       max_input_len=max_input_len or self.max_input_len,
                       max_gen_len=max_gen_len or self.max_gen_len,
                       zero_counts=zero_counts,
                       steps=steps,
                       check_state=check_state,
                       layer_indices=layer_indices,
                       info=info,
                       mode=mode,
                       loss_coefs=loss_coefs,
                       ds=ds)
        self.optimizer.zero_grad(set_to_none=False)
        self.optimizer = None
        self.shuffle_sample_indices(ds=ds)
        return ret

    def train_model(self,
                    bs=8,
                    mbs=8,
                    pbs=8,
                    max_input_len=None,
                    max_gen_len=None,
                    steps=1024,
                    lr=0.0003,
                    lrs='cos',
                    optim='SGD',
                    layer_indices=None,
                    check_state=None,
                    mode='full',
                    loss_coefs=None,
                    checkpoint=False,
                    info='',
                    ds='sft'):
        assert mode in ('full', 'block', 'upper', 'lower')
        assert lrs in ('cos', 'fix')
        assert ds in ('sft', 'aux')

        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=False)
        self.model.train()
        if layer_indices is None:
            layer_indices = range(self.n_layer)

        self.log(f'\n{mode=} {bs=} {mbs=} {steps=} {lr=} {lrs=} {optim=} {loss_coefs=} {ds=}')

        # no embedding layer
        trainable_params = []
        if mode == 'full' or mode == 'upper':
            trainable_params.extend(self.get_head().parameters())
            if self.get_final_norm() is not None:
                trainable_params.extend(self.get_final_norm().parameters())
            if hasattr(self.model, 'advantage'):
                trainable_params.extend(self.model.advantage.parameters())

        if mode == 'full' or mode == 'lower':
            trainable_params.extend(self.get_wte().parameters())

        for index in layer_indices:
            if index >= self.n_layer:
                continue
            for n, p in self.get_layers()[index].named_parameters():
                if 'mask' in n:
                    continue
                trainable_params.append(p)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in trainable_params:
            p.requires_grad = True

        trainable_params = list(set(trainable_params))
        if optim == 'SGD':
            self.optimizer = SGD(trainable_params, lr=lr * mbs / bs)
        elif optim == 'Adam':
            self.optimizer = Adam(trainable_params, lr=lr * mbs / bs, betas=(0.9, 0.95))
        elif optim == 'AdamW':
            self.optimizer = AdamW(trainable_params, lr=lr * mbs / bs, betas=(0.9, 0.95), weight_decay=0.1)
        self.trainable_params = trainable_params
        if lrs == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        51200 // bs,
                                                                        eta_min=0,
                                                                        last_epoch=-1,
                                                                        verbose=False)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1024, gamma=1.0)
        self.get_transformer().gradient_checkpointing = checkpoint
        ret = self.fit(bs=bs,
                       mbs=mbs,
                       pbs=pbs,
                       max_input_len=max_input_len or self.max_input_len,
                       max_gen_len=max_gen_len or self.max_gen_len,
                       steps=steps,
                       check_state=check_state,
                       layer_indices=layer_indices,
                       mode=mode,
                       loss_coefs=loss_coefs,
                       info=info,
                       ds=ds)
        self.optimizer.zero_grad(set_to_none=False)
        self.get_transformer().gradient_checkpointing = False
        self.optimizer = None
        self.shuffle_sample_indices(ds=ds)
        return ret

    def fit(self,
            bs=4,
            mbs=4,
            pbs=4,
            max_input_len=256,
            max_gen_len=64,
            zero_counts=None,
            steps=None,
            check_state=None,
            layer_indices=None,
            info='',
            mode='mlp',
            loss_coefs=None,
            ds='sft'):
        assert ds in ('sft', 'aux')
        assert mode in ('mlp', 'attn', 'dim', 'depth', 'full', 'block', 'upper', 'lower')
        stage = None
        if mode in ('depth',):
            stage = 'depth'
            if zero_counts is None:
                zero_counts = [(i)/steps for i in range(steps)]
            assert steps is not None
        elif mode in ('mlp', 'attn', 'dim'):
            stage = 'layer'
            assert zero_counts is not None
            steps = len(zero_counts)
        elif mode in ('full', 'block', 'upper', 'lower'):
            stage = 'model'
            assert steps is not None

        losses = []
        d_losses = []
        e_losses = []
        a_losses = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        queries = self.queries if ds == 'sft' else self.aux_queries
        answers = self.answers if ds == 'sft' else self.aux_answers
        embeddings = self.embeddings if ds == 'sft' else self.aux_embeddings
        logit_cache = self.logit_cache if ds == 'sft' else self.aux_logit_cache
        sample_indices = self.sample_indices if ds == 'sft' else self.aux_sample_indices
        assert len(sample_indices) > 0

        input_batch_count = max(len(sample_indices) // bs, 1)
        ts = time.time()
        i = 0
        ppl = 1e6
        acc = 0.0
        for ic in range(steps):
            if i >= input_batch_count:
                self.shuffle_sample_indices(ds=ds)
                i = i % input_batch_count

            indices = sample_indices[i * bs:(i + 1) * bs]
            qs = [queries[x] for x in indices]
            rs = [answers[x] for x in indices]
            cs = logit_cache[indices] if logit_cache is not None else None
            embs = embeddings[indices] if embeddings is not None else None

            loss, d_loss, e_loss, a_loss = self._calc_loss(qs,
                                                           rs,
                                                           cs,
                                                           embeddings=embs,
                                                           max_input_len=max_input_len,
                                                           max_gen_len=max_gen_len,
                                                           mbs=mbs,
                                                           loss_coefs=loss_coefs)
            if stage == 'depth':
                mask_counts, topk_values, max_grads = self._update_mask(mode=mode,
                                                                        zero_count=zero_counts[ic])
            elif stage == 'layer':
                zero_count = zero_counts[ic]
                max_zero_count = max(zero_counts)
                mask_counts, topk_values, max_grads = self._update_mask(mode=mode,
                                                                        zero_count=zero_count,
                                                                        max_zero_count=max_zero_count,
                                                                        layer_indices=layer_indices)
            torch.nn.utils.clip_grad_norm_(parameters=self.trainable_params, max_norm=len(queries) / mbs)
            self.optimizer.step()
            self.scheduler.step()

            if torch.cuda.is_available():
                mem = round(torch.cuda.memory_reserved() / 1024 ** 3, 3)
            else:
                mem = 0.0

            i += 1
            losses.append(loss)
            d_losses.append(d_loss)
            e_losses.append(e_loss)
            a_losses.append(a_loss)

            if (ic + 1) % self.log_steps == 0 or ic == steps - 1:
                b = self.log_steps
                mean_loss = sum(losses[-b:]) / len(losses[-b:])
                mean_d_loss = sum(d_losses[-b:]) / len(d_losses[-b:])
                mean_e_loss = sum(e_losses[-b:]) / len(e_losses[-b:])
                mean_a_loss = sum(a_losses[-b:]) / len(a_losses[-b:])

                ppl = np.exp(mean_e_loss)
                if stage == 'layer' or stage == 'depth':
                    max_mask_count = max(mask_counts)
                    mean_values = sum(topk_values) / max(len(topk_values), 1)
                    max_grad = max(max_grads)
                else:
                    max_mask_count = -1
                    mean_values = -1
                    max_grad = -1
                min_idx = min(layer_indices) if layer_indices else 0
                max_idx = max(layer_indices) if layer_indices else 0
                n_layer = self.n_layer
                elapse = time.time() - ts
                self.log(
                    f'\n{info} {stage}:{mode} layers:{min_idx}-{max_idx}/{n_layer} ds:{ds} bs:{mbs}/{bs} '
                    f'step:{ic + 1}/{steps} loss:{mean_loss:.3f} '
                    f'kl_loss:{mean_d_loss:.3f} emp_loss:{mean_e_loss:.3f} aux_loss:{mean_a_loss:.3f} '
                    f'ppl:{ppl:.3f} mask.count:{max_mask_count:.2f} '
                    f'topk.values:{mean_values:.3g} grad.max:{max_grad:.3f} mem:{mem:.3f} time:{elapse:.3f}')
                if (ic + 1) % self.eval_steps == 0 or ic == steps - 1 or stage == 'layer' and max(mask_counts) >= max(zero_counts):
                    if ic == steps - 1 or stage == 'layer' and max(mask_counts) >= max(zero_counts):
                        count = None
                    elif (ic + 1) % (100*self.eval_steps) == 0:
                        count = 100 * self.eval_count
                    elif (ic + 1) % (10*self.eval_steps) == 0:
                        count = 10 * self.eval_count
                    else:
                        count = self.eval_count
                    line = f'{info} {stage}:{mode} layers:{min_idx}-{max_idx}/{n_layer} step:{ic + 1}/{steps}'
                    acc = self.calc_acc(count=count,
                                             batch_size=pbs,
                                             max_log_count=self.error_count,
                                             info=line)
                    if stage == 'layer' and max(mask_counts) >= max(zero_counts):
                        return {"suc": True, "ppl": ppl, "acc": acc}

                if check_state is not None:
                    check_steps = check_state[0]
                    val = acc  # TODO: -ppl
                    min_val = check_state[1]
                    max_val = check_state[2]
                    if ic >= check_steps and val < min_val and stage == 'layer':
                        return {"suc": False, "ppl": ppl, "acc": acc}
                    elif ic >= check_steps and val > max_val and stage in ('depth', 'model'):
                        return {"suc": True, "ppl": ppl, "acc": acc}

        return {"suc": True, "ppl": ppl, "acc": acc}

    def forward(self, input_ids, position_ids=None, attention_mask=None, embeddings=None):
        raise NotImplementedError('forward')

    def _calc_loss(self,
                   queries,
                   answers,
                   caches,
                   embeddings=None,
                   max_input_len=256,
                   max_gen_len=64,
                   loss_coefs=None,
                   mbs=8):
        inputs = self.tokenize(queries, answers, max_input_len=max_input_len, max_gen_len=max_gen_len)

        t_hs = None
        if caches is not None:
            t_hs = torch.from_numpy(caches).to(self.train_dtype).to(self.device)
        if embeddings is not None:
            embeddings = torch.from_numpy(embeddings).to(self.train_dtype).to(self.device)

        labels = inputs['labels']
        input_ids = inputs['input_ids']
        position_ids = inputs.get('position_ids', None)
        attention_mask = inputs.get('attention_mask', None)

        # emp_ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        emp_ce = nn.CrossEntropyLoss(reduction='none')
        dis_ce = nn.CrossEntropyLoss(reduction='none')

        loss_val = 0.0
        kl_loss_val = 0.0
        emp_loss_val = 0.0
        aux_loss_val = 0.0
        self.optimizer.zero_grad(set_to_none=False)
        nb = max(len(queries) // mbs, 1)

        for ib in range(nb):

            labels_ = labels[ib * mbs:(ib + 1) * mbs, max_input_len:]
            label_mask = (labels_ != -100).to(torch.float).reshape(-1)
            label_mask_count = torch.sum(label_mask)

            input_ids_ = input_ids[ib * mbs:(ib + 1) * mbs]
            position_ids_ = position_ids[ib * mbs:(ib + 1) * mbs] if position_ids is not None else None
            attention_mask_ = attention_mask[ib * mbs:(ib + 1) * mbs] if attention_mask is not None else None
            embeddings_ = embeddings[ib * mbs:(ib + 1) * mbs] if embeddings is not None else None
            s_hs = self.forward(input_ids=input_ids_,
                                position_ids=position_ids_,
                                attention_mask=attention_mask_,
                                embeddings=embeddings_)
            s_hs = s_hs[:, max_input_len:]

            if hasattr(self.model, 'mask') and self.model.mask is not None:
                s_hs = s_hs * self.model.mask

            s_logits = self.get_head()(s_hs).view(-1, self.n_voc).float()

            emp_losses = emp_ce(s_logits, labels_.reshape(-1))
            emp_loss = torch.sum(emp_losses * label_mask) / label_mask_count
            emp_loss_val += emp_loss.clone().detach().item()

            loss = 0.0
            if 'emp' in loss_coefs:
                loss = loss + loss_coefs['emp'] * emp_loss

            if 'kl' in loss_coefs or 'pair' in loss_coefs or 'ppo' in loss_coefs:
                fixed_head_weight = self.model.fixed_head_weight
                t_logits = F.linear(t_hs[ib * mbs:(ib + 1) * mbs], fixed_head_weight).view(-1, self.n_voc).float()
                t_probs = t_logits.softmax(dim=1).detach()

                if 'kl' in loss_coefs:
                    kl_losses = dis_ce(s_logits, t_probs)
                    kl_loss = torch.sum(kl_losses * label_mask) / label_mask_count
                    kl_loss_val += kl_loss.clone().detach().item()
                    loss = loss + loss_coefs['kl'] * kl_loss

                if 'pair' in loss_coefs:
                    zero = torch.zeros((), dtype=torch.long).cuda()
                    s_label_logits = torch.gather(s_logits, 1,
                                                  torch.maximum(labels_, zero).reshape(-1, 1)).squeeze(1)
                    top_logits = torch.mean(torch.topk(s_logits, 2, dim=1, largest=True).values, dim=1)
                    coef = torch.minimum(torch.abs(top_logits - s_label_logits),
                                         torch.ones((), dtype=torch.float).to(self.device))
                    aux_losses = torch.maximum((top_logits - s_label_logits) * coef,
                                               torch.zeros((), dtype=torch.float).to(self.device))
                    aux_loss = torch.sum(aux_losses * label_mask) / label_mask_count
                    aux_loss_val += aux_loss.clone().detach().item()
                    loss = loss + loss_coefs['pair'] * aux_loss

            loss.backward(retain_graph=False)
            loss_val += loss.clone().detach().item()

        return loss_val / nb, kl_loss_val / nb, emp_loss_val / nb, aux_loss_val / nb

    def _update_mask(self, mode='mlp', zero_count=16, max_zero_count=1024, layer_indices=None):
        if layer_indices is None:
            layer_indices = range(self.n_layer)
        mask_counts = []
        topk_values = []
        max_grads = []

        if mode == 'mlp':
            for index in layer_indices:
                kernel = getattr(self.get_layers()[index], self.mlp_name)
                mask_counts_, topk_values_, max_grads_ = kernel.update_mask(zero_count=zero_count,
                                                                            max_zero_count=max_zero_count)
                mask_counts.extend(mask_counts_)
                topk_values.extend(topk_values_)
                max_grads.extend(max_grads_)
            self.mask_counts['mlp'] = sum(mask_counts)/len(mask_counts)
        elif mode == 'attn':
            for index in layer_indices:
                kernel = getattr(self.get_layers()[index], self.attn_name)
                mask_counts_, topk_values_, max_grads_ = kernel.update_mask(zero_count=zero_count,
                                                                            max_zero_count=max_zero_count)
                mask_counts.extend(mask_counts_)
                topk_values.extend(topk_values_)
                max_grads.extend(max_grads_)
            self.mask_counts['attn'] = sum(mask_counts)/len(mask_counts)
        elif mode == 'dim':
            amp = 0.0
            for index in layer_indices:
                kernel = self.get_layers()[index]
                amp_, max_grads_ = kernel.calc_sensitive()
                amp += amp_
                max_grads.extend(max_grads_)

            topk = torch.topk(torch.abs(amp), max(zero_count, 1), largest=False)
            self.model.mask[topk.indices] = 0.0
            mask_counts.append((1 - self.model.mask).long().sum().item())
            topk_values.append(torch.mean(topk.values).item())
            self.mask_counts['dim'] = mask_counts[0]
        elif mode == 'depth':
            kernel = self.get_layers()[-1]
            kernel.update_mask(zero_count=zero_count)
            mask_counts = [kernel.mask.item()]
            topk_values = [-1.0]
            max_grads = [-1.0]
        else:
            mask_counts = [-1.0]
            topk_values = [-1.0]
            max_grads = [-1.0]

        return mask_counts, topk_values, max_grads

    def clip_layer(self, clip_index=None):
        if clip_index is None:
            self.set_layers(self.get_layers()[:-1])
        else:
            self.set_layers(self.get_layers()[:clip_index])

    def load_sample(self, count=None, ds='sft'):
        assert ds in ('sft', 'aux')
        queries = []
        answers = []
        preds = []
        sample_dir = self.sample_dir if ds == 'sft' else self.aux_sample_dir
        ts = time.time()
        for i, line in enumerate(open(sample_dir, 'r')):
            line = json.loads(line)

            q = line["prompt"]
            a = line['response']
            p = ''

            queries.append(q)
            answers.append(a)
            preds.append(p)
            if count is not None and len(queries) >= count:
                break

        emb_dir = self.emb_dir if ds == 'sft' else self.aux_emb_dir
        if emb_dir is not None:
            with open(emb_dir, 'rb') as f:
                embeddings = np.load(f)
                if count is not None:
                    embeddings = embeddings[:count]
            assert len(queries) == len(embeddings)
        else:
            embeddings = None

        te = time.time()
        self.log(f'load sample from {sample_dir} count:{len(queries)} time:{te - ts:.2f}s')

        if ds == 'sft':
            self.queries = queries
            self.answers = answers
            self.embeddings = embeddings
            self.sample_indices = list(range(len(self.queries)))
        else:
            self.aux_queries = queries
            self.aux_answers = answers
            self.aux_embeddings = embeddings
            self.aux_sample_indices = list(range(len(self.aux_queries)))

    def split_sft_sample(self, rate_or_count=0.05, scatter=False):
        count = len(self.sample_indices)
        if 0 < rate_or_count < 1:
            test_count = int(rate_or_count * count)
        else:
            test_count = rate_or_count

        if scatter:
            # get scatter test samples
            random.Random(7).shuffle(self.sample_indices)

        self.test_queries = self.queries[:test_count]
        self.test_answers = self.answers[:test_count]
        if self.embeddings is not None:
            self.test_embeddings = self.embeddings[:test_count]
        if self.logit_cache is not None:
            self.test_logit_cache = self.logit_cache[:test_count]

        self.sample_indices = list(self.sample_indices[test_count:])

    def shrink_sft_sample(self, rate_or_count=0.10, scatter=False):
        # keep rate_or_count samples
        count = len(self.sample_indices)
        if 0 < rate_or_count < 1:
            keep_count = int(rate_or_count * count)
        else:
            keep_count = rate_or_count

        if scatter:
            # get scatter test samples
            random.Random(7).shuffle(self.sample_indices)

        self.sample_indices = self.sample_indices[:keep_count]

    def aux_to_test(self):
        self.test_queries = self.aux_queries
        self.test_answers = self.aux_answers
        if self.aux_embeddings is not None:
            self.test_embeddings = self.aux_embeddings
        self.test_logit_cache = self.aux_logit_cache
        self.test_sample_indices = self.aux_sample_indices

    def shuffle_sample_indices(self, ds='sft'):
        assert ds in ('sft', 'aux')
        if ds == 'sft':
            random.Random(7).shuffle(self.sample_indices)
        else:
            random.Random(7).shuffle(self.aux_sample_indices)

    def sort_test_samples(self, indices):
        self.test_queries = [self.test_queries[x] for x in indices]
        self.test_answers = [self.test_answers[x] for x in indices]
        if self.test_embeddings is not None:
            self.test_embeddings = self.test_embeddings[indices]
        if self.test_logit_cache is not None:
            self.test_logit_cache = self.test_logit_cache[indices]

    def replace_kernels(self, mode='mlp', layer_indices=None, mask=None):
        kernels = []

        if mode == 'depth':
            block = self.get_layers()[-1]
            patched = isinstance(block, SparseBlock)
            if not patched:
                kernel = SparseBlock(layer=block, layer_index=self.n_layer - 1)
                self.get_layers()[-1] = kernel
                kernels.append(kernel)
            else:
                kernels.append(block)
            return kernels

        for i in layer_indices:
            if mode == 'mlp':
                mlp = getattr(self.get_layers()[i], self.mlp_name)
                patched = isinstance(mlp, SparseMLP)
                if not patched:
                    kernel = self._get_sparse_mlp(layer=mlp, layer_index=i)
                    setattr(self.get_layers()[i], self.mlp_name, kernel)
                    kernels.append(kernel)
                else:
                    kernels.append(mlp)
            elif mode == 'attn':
                attn = getattr(self.get_layers()[i], self.attn_name)
                patched = isinstance(attn, SparseAttn)
                if not patched:
                    kernel = self._get_sparse_attn(layer=attn, layer_index=i)
                    setattr(self.get_layers()[i], self.attn_name, kernel)
                    kernels.append(kernel)
                else:
                    kernels.append(attn)
            elif mode == 'dim':
                block = self.get_layers()[i]
                patched = isinstance(block, SparseDim)
                if not patched:
                    kernel = self._get_sparse_dim(layer=block, layer_index=i, mask=mask)
                    mlp = kernel.get(kernel.mlp_name)
                    if not isinstance(mlp, SparseMLP):
                        kernel.set(kernel.mlp_name, self._get_sparse_mlp(layer=mlp, layer_index=i))
                    attn = kernel.get(kernel.attn_name)
                    if not isinstance(attn, SparseAttn):
                        kernel.set(kernel.attn_name, self._get_sparse_attn(layer=attn, layer_index=i))
                    self.get_layers()[i] = kernel
                    kernels.append(kernel)
                else:
                    kernels.append(block)
        return kernels

    def get_params(self, kernels):
        params = set()
        for k in kernels:
            for n, p in k.named_parameters():
                if 'mask' in n:
                    continue
                params.add(p)
        return list(params)

    def reparam(self, mode='mlp'):
        assert mode in ('mlp', 'attn', 'dim')
        for i, h in enumerate(self.get_layers()):
            if mode == 'mlp':
                kernel = getattr(h, self.mlp_name)
                if not isinstance(kernel, SparseMLP):
                    continue
                kernel.reparam()
            elif mode == 'attn':
                kernel = getattr(h, self.attn_name)
                if not isinstance(kernel, SparseAttn):
                    continue
                kernel.reparam()
            elif mode == 'dim':
                if not isinstance(h, SparseDim):
                    continue
                h.reparam()

    def get_layers(self):
        # return self.model.model.layers
        raise NotImplementedError('get_layers')

    def set_layers(self, layers):
        # self.model.model.layers = layers
        raise NotImplementedError('set_layers')

    def get_final_norm(self):
        # return self.model.model.norm
        raise NotImplementedError('get_final_norm')

    def set_final_norm(self, norm):
        # self.model.model.norm = norm
        raise NotImplementedError('set_final_norm')

    def get_transformer(self):
        # return self.model.model
        raise NotImplementedError('get_transformer')

    def set_transformer(self, transformer):
        # self.model.model = transformer
        raise NotImplementedError('set_transformer')

    def get_head(self):
        # return self.model.lm_head
        raise NotImplementedError('get_head')

    def set_head(self, head):
        # self.model.lm_head = head
        raise NotImplementedError('set_head')

    def get_wte(self):
        # return self.model.model.embed_tokens
        raise NotImplementedError('get_wte')

    def set_wte(self, wte):
        # self.model.model.wte = wte
        raise NotImplementedError('set_wte')

    def get_wpe(self):
        # return None
        raise NotImplementedError('get_wpe')

    def set_wpe(self, wpe):
        # pass
        raise NotImplementedError('set_wpe')

    @property
    def n_layer(self):
        # return len(self.model.model.layers)
        raise NotImplementedError('set_wpe')

    def eval(self):

        self._mlp_eval()

        self._attn_eval()

        if hasattr(self.model, 'mask') and self.model.mask is not None:
            self._dim_eval()

    def _mlp_eval(self):
        mlp_idx = []
        attn_idx = []
        dim_idx = []

        for i, layer in enumerate(self.get_layers()):
            kernel = getattr(layer, self.mlp_name)
            if isinstance(kernel, SparseMLP):
                mlp_idx.append(str(i))
                kernel.eval()
        self.log(f'set mlp to eval mode:{",".join(mlp_idx)}')

    def _attn_eval(self):
        attn_idx = []
        for i, layer in enumerate(self.get_layers()):
            kernel = getattr(layer, self.attn_name)
            if isinstance(kernel, SparseAttn):
                attn_idx.append(str(i))
                kernel.eval()
        self.log(f'set attn to eval mode:{",".join(attn_idx)}')

    def _dim_eval(self):

        dim_idx = []
        for i, layer in enumerate(self.get_layers()):
            if isinstance(layer, SparseDim):
                dim_idx.append(str(i))
                layer.eval()

        self.log(f'set dim to eval mode:{",".join(dim_idx)}')

        mask = self.model.mask.data == 1.0

        wte = self.get_wte()
        wte.weight = Parameter(
            wte.weight[:, mask].contiguous().detach(), requires_grad=False)
        wte.embedding_dim = wte.weight.size(1)
        wte.training = False

        wpe = self.get_wpe()
        if wpe is not None:
            if not isinstance(wpe, (list, tuple)):
                wpe = [wpe]
            for emb in wpe:
                emb.weight = Parameter(
                    emb.weight[:, mask].contiguous().detach(), requires_grad=False)
                emb.training = False
                emb.embedding_dim = emb.weight.size(1)

        head = self.get_head()
        head.weight = Parameter(head.weight[:, mask].contiguous().detach(),
                                requires_grad=False)
        head.in_features = head.weight.size(1)
        head.training = False

        final_norm = self.get_final_norm()
        if final_norm is not None:
            final_norm.eval()

        self.input_emb_mask = self.model.mask

        squeeze_mask = Parameter(self.model.mask[self.model.mask == 1], requires_grad=False)
        self.model.mask = squeeze_mask

    def train(self):
        for i, layer in enumerate(self.get_layers()):
            getattr(layer, self.mlp_name).training = True
            getattr(layer, self.attn_name).training = True
            getattr(layer, self.ln1_name).training = True
            getattr(layer, self.ln2_name).training = True
            layer.training = True
        self.get_final_norm().training = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def count_params(self, head=True):
        ps = []
        if head:
            for p in self.get_head().parameters():
                if len(p.shape) > 1:
                    ps.append(reduce(lambda x, y: x * y, p.shape))
                elif len(p.shape) == 1:
                    ps.append(p.shape[0])
            for p in self.get_wte().parameters():
                if len(p.shape) > 1:
                    ps.append(reduce(lambda x, y: x * y, p.shape))
                elif len(p.shape) == 1:
                    ps.append(p.shape[0])
        for m in self.get_layers():
            for p in m.parameters():
                if len(p.shape) > 1:
                    ps.append(reduce(lambda x, y: x * y, p.shape))
                elif len(p.shape) == 1:
                    ps.append(p.shape[0])
        return sum(ps)

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

    def calc_logit(self, ds='sft', coef=None, bs=32):
        assert ds in ('sft', 'aux')
        ts = time.time()
        if ds == 'sft':
            self.logit_cache = None
            queries, answers, embeddings, logit_dir = self.queries, self.answers, self.embeddings, self.logit_dir
        else:
            self.aux_logit_cache = None
            queries, answers, embeddings, logit_dir = self.aux_queries, self.aux_answers, self.aux_embeddings, self.aux_logit_dir

        logit_cache = self._batch_calc_logits(queries, answers, embeddings=embeddings, bs=bs)

        if ds == 'sft':
            self.logit_cache = logit_cache
        else:
            self.aux_logit_cache = logit_cache

        with open(logit_dir, 'wb') as f:
            np.savez(f, logits=logit_cache)
        self.log(f'save {ds} logits in {round(time.time() - ts, 3)}s')

    def _batch_calc_logits(self, queries, answers, embeddings=None, bs=16):
        length = len(queries)
        d = self.get_wte().weight.size(1)
        caches = np.zeros([length, self.max_gen_len, d], dtype=np.float16)

        ts = time.time()
        steps = (length - 1) // bs + 1
        for i in range(steps):
            if (i + 1) % 10 == 0:
                self.log(f'step:{i + 1}/{steps} bs:{bs} elapse:{round(time.time() - ts, 2)}s')
            with torch.no_grad():
                pre_logit = self._calc_logits(queries[i * bs:(i + 1) * bs],
                                              answers[i * bs:(i + 1) * bs],
                                              embeddings=embeddings[
                                                         i * bs:(i + 1) * bs] if embeddings is not None else None
                                              )
                pre_logit = pre_logit.half().detach().cpu()
                caches[i * bs:(i + 1) * bs] = pre_logit

        return caches

    def _calc_logits(self, queries, answers, embeddings=None):
        inputs = self.tokenize(queries, answers, max_input_len=self.max_input_len, max_gen_len=self.max_gen_len)
        if embeddings is not None:
            embeddings = torch.from_numpy(embeddings).half().to(self.device)
        pre_logits = self.forward(input_ids=inputs['input_ids'],
                                  position_ids=inputs.get('position_ids', None),
                                  attention_mask=inputs['attention_mask'],
                                  embeddings=embeddings
                                  )

        pre_logit = pre_logits[:, self.max_input_len:]
        return pre_logit

    def load_logit(self, count=None, ds='sft'):
        ts = time.time()
        assert ds in ('sft', 'aux')
        logit_dir = self.logit_dir if ds == 'sft' else self.aux_logit_dir
        with open(logit_dir, 'rb') as f:
            caches = np.load(f)
            logit_cache = caches['logits']
            if count is not None:
                logit_cache = logit_cache[:count]
            avg_logit_cache = caches.get('avg_logits', None)
            if avg_logit_cache is not None and count is not None:
                avg_logit_cache = avg_logit_cache[:count]

        if ds == 'sft':
            assert len(logit_cache) == len(self.sample_indices), f'{len(logit_cache)=} {len(self.sample_indices)=}'
            self.logit_cache = logit_cache
            self.avg_logit_cache = avg_logit_cache
        else:
            assert len(logit_cache) == len(
                self.aux_sample_indices), f'{len(logit_cache)=} {len(self.aux_sample_indices)=}'
            self.aux_logit_cache = logit_cache
            self.aux_avg_logit_cache = avg_logit_cache
        self.log(f'load {ds} logits in {round(time.time() - ts, 3)}s')

    def mock_logit(self, count=None, ds='sft'):
        ts = time.time()
        assert ds in ('sft', 'aux')
        if ds == 'sft':
            n_sample = len(self.sample_indices)
            self.logit_cache = np.zeros((n_sample, self.max_gen_len, self.get_wte().weight.size(1)), dtype=np.float16)
        else:
            n_sample = len(self.aux_sample_indices)
            self.aux_logit_cache = np.zeros((n_sample, self.max_gen_len, self.get_wte().weight.size(1)),
                                            dtype=np.float16)
        self.log(f'mock {ds} logits in {round(time.time() - ts, 3)}s')

    def tokenize(self, queries=None, answers=None, max_input_len=256, max_gen_len=128):
        bos = 1
        eos = 2
        pad = 2
        if answers is not None:

            source_inputs = self.tokenizer(queries, max_length=max_input_len, padding='max_length', truncation=True)
            source_ids = source_inputs.input_ids
            source_mask = source_inputs.attention_mask

            target_inputs = self.tokenizer(answers, max_length=max_gen_len, padding='max_length', truncation=True)
            target_ids = target_inputs.input_ids
            target_mask = target_inputs.attention_mask

            input_ids = []
            label_ids = []
            attention_mask = []
            for i, target_ids_ in enumerate(target_ids):
                source_ids_ = source_ids[i]
                input_ids.append(source_ids_ + target_ids_)
                # mask bos in target ids
                index = target_mask[i].index(1)
                target_mask_ = target_mask[i]
                target_mask_[index] = 0
                # index==0: exceed max length, last token is not eos, so mask the last one
                suffix = -100 if index == 0 else eos
                attention_mask.append(source_mask[i] + target_mask_)
                label_ids_ = [-100] * max_input_len + [-100 if x == pad or x == bos else x for x in target_ids_[1:]] + [
                    suffix]
                label_ids.append(label_ids_)
            input_ids = torch.tensor(input_ids).long().to(self.device)
            label_ids = torch.tensor(label_ids).long().to(self.device)
            attention_mask = torch.tensor(attention_mask).long().to(self.device)

            # position_ids = attention_mask.cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask == 0, 1)

            batch = {}
            batch["input_ids"] = input_ids
            batch["labels"] = label_ids
            batch["attention_mask"] = attention_mask
            batch["position_ids"] = None
            return BatchEncoding(batch)

        else:
            # inputs = self.tokenizer(queries, return_tensors="pt")
            inputs = self.tokenizer(queries, max_length=max_input_len,
                                    padding='max_length', truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            # position_ids = attention_mask.cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask == 0, 1)

            batch = {}
            batch["input_ids"] = input_ids
            batch["attention_mask"] = attention_mask
            batch["position_ids"] = None
            return BatchEncoding(batch)

    def batch_chat(self, qs, ans=None, embeddings=None, max_length=1024, batch_size=1, log_query=False, max_log_count=0,
                   log_truth=False,
                   log_answer=True, log_summary=False, log_space=False, emit=False):
        total_out_tokens = 1e-6
        total_times = 1e-6

        self.get_transformer().gradient_checkpointing = False

        rs = []
        for i in range((len(qs) - 1) // batch_size + 1):
            queries = qs[i * batch_size:(i + 1) * batch_size]
            answers = ans[i * batch_size:(i + 1) * batch_size] if ans else None
            embeddings_ = embeddings[i * batch_size:(i + 1) * batch_size] if embeddings is not None else None

            speeds = []

            in_char = 0
            in_token = 0
            out_char = 0
            out_token = 0
            ts = time.time()
            input_texts, input_ids, masks, output_ids, output_texts = self.chat(queries,
                                                                                embeddings=embeddings_,
                                                                                max_length=max_length)
            rs.extend(output_texts)
            in_char += sum([len(x) for x in queries])
            in_token += sum([len(x) for x in input_ids])
            out_char += sum([len(x) for x in output_texts])
            out_token += sum([len(x) for x in output_ids])
            t = (time.time() - ts)
            speed_char = out_char / t
            speed_token = out_token / t
            speeds.append(speed_token)
            total_out_tokens += out_token
            total_times += t
            if log_space:
                self.log('')
            for j, q in enumerate(queries):
                if i * batch_size + j >= max_log_count:
                    continue
                if log_query:
                    # q = q.replace('\n', '\\n')
                    self.log(f"****Prompt****:{q}")
                if log_truth:
                    self.log(f"****Truth****:{answers[j]}")
                if log_answer:
                    self.log(f"****Pred*****:{output_texts[j]}")
                if log_space:
                    self.log('')
            if log_summary:
                self.log(
                    f"{i}/{len(qs)} "
                    f"input:{in_char:.1f}/{in_token:.1f} output:{out_char:.1f}/{out_token:.1f} "
                    f"time:{t:.3f}s speed:{speed_char:.1f}/{speed_token:.1f}")
            if log_space:
                self.log('')
        if emit:
            return rs

    def chat(self, queries, embeddings=None, max_length=256, strip=True):
        inputs = self.tokenize(queries, max_input_len=self.max_input_len, max_gen_len=max_length)
        # print(f'{queries=} {inputs=}')
        input_ids = inputs['input_ids']
        input_length = input_ids.shape[1]
        attention_mask = inputs['attention_mask']
        if embeddings is not None:
            embeddings = torch.from_numpy(embeddings)
            if self.input_emb_mask is not None:
                embeddings = embeddings[:,:, self.input_emb_mask==1.0]
            emb_kwargs = {'inputs_embeds': embeddings}
        else:
            emb_kwargs = {}
        outputs = self.model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      position_ids=inputs['position_ids'],
                                      use_cache=True,
                                      max_length=max_length + input_length,
                                      repetition_penalty=1.0,
                                      do_sample=False,
                                      temperature=1.0,
                                      pad_token_id=2,
                                      #  top_k=10,
                                      #  num_beams=1,
                                      #  top_p=0.9,
                                      # num_beams=3,
                                      # num_return_sequences=1,
                                      **emb_kwargs
                                      )
        io_ids = outputs.tolist()
        input_ids = [x[:input_length] for x in io_ids]
        output_ids = [x[input_length:] for x in io_ids]
        input_texts = []
        output_texts = []
        for i, o_ids in enumerate(output_ids):
            # if 2 in o_ids:
            #     o_ids = token_ids[:o_ids.index(2) + 1]
            input_text = self.tokenizer.decode(input_ids[i])
            input_texts.append(input_text)
            output_text = self.tokenizer.decode(o_ids)
            if strip:
                output_text = output_text.replace("</s>", "").replace("<s>", "").strip()
            output_texts.append(output_text)
        return input_texts, input_ids, attention_mask, output_ids, output_texts

    def save_conf(self, save_dir=None, prefix='LlamaConfig'):
        raise NotImplementedError('save_conf')

    def copy_conf(self, src_dir=None, dst_dir=None):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src_dir + '/config.json', dst_dir + '/config.json')

    def save_model(self, save_dir=None, dtype=torch.float16):
        ts = time.time()
        sd = {k: x.to(dtype) for k, x in self.model.state_dict().items() if 'fixed_head_weight' not in k}
        os.makedirs(save_dir, exist_ok=True)
        torch.save(sd, save_dir + '/pytorch_model.bin')
        if hasattr(self, 'input_emb_mask') and self.input_emb_mask is not None:
            mask = self.input_emb_mask
            with open(save_dir + '/mask.npz', 'wb') as f:
                np.savez(f, mask=mask.long().cpu().numpy())
        self.log(f'save model:{save_dir.split(",")[-1]} in {round(time.time() - ts, 3)}s')

    def save_ckpt(self, save_dir=None, dtype=torch.bfloat16):
        ts = time.time()
        sd = {k: x.to(dtype) for k, x in self.model.state_dict().items()}
        os.makedirs(save_dir, exist_ok=True)
        torch.save(sd, save_dir + '/pytorch_model.bin')
        self.log(f'save model:{save_dir.split(",")[-1]} in {round(time.time() - ts, 3)}s')

    def log(self, s):
        print(s)
        if self.logger is not None:
            self.logger.write(s + '\n')
            self.logger.flush()

    def star_log(self, s):
        s = f'\n{"*" * 64} {s} {"*" * 64}\n'
        self.log(s)

    def profile(self, queries):
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()
        self.batch_chat(queries)
        pr.disable()
        s = io.StringIO()
        # sortby = SortKey.CUMULATIVE SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME)
        ps.print_stats(32)
        print(s.getvalue())

    def calc_acc(self, count=100, batch_size=1, ds='test', max_log_count=0, filename=None, info=''):
        assert ds in ('test', 'train', 'aux')
        count = 1000000 if count is None else count

        ts = time.time()
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
        for i, ans in enumerate(rs):
            pred = ans.replace('<s>', '').replace('</s>', '').strip()
            true = answers[i].replace('<s>', '').replace('</s>', '').strip()
            n += 1
            score = scorer.score(prediction=pred, target=true)["rougeL"].fmeasure
            n_p += score
            n_ps[i // 100] += score
        acc = n_p / n
        details = [round(x / 100, 3) for x in n_ps]
        self.log(f'\nrouge:{acc:.3f} sample:{n} details:{details} elapse:{te - ts:.3f} {info}\n')
        return acc


class DistillPipe():
    def __init__(self,
                 worker=None,
                 sample_count=None,
                 bs=1024,
                 mbs=2,
                 pbs=16,
                 itv=1,
                 optim='AdamW',
                 lr=0.001,
                 lrs='cos',
                 train_stages=1,
                 layer_loss_coefs=None,
                 model_loss_coefs=None,
                 check_state=None,
                 token_dir=None,
                 conf_dir=None,
                 save_dir=None,
                 save_stages=None):
        self.worker = worker
        self.sample_count = sample_count
        self.bs = bs
        self.mbs = mbs
        self.pbs = pbs
        self.itv = itv
        self.optim = optim
        self.lr = lr
        self.lrs = lrs
        self.train_stages = train_stages
        self.layer_loss_coefs = layer_loss_coefs
        self.model_loss_coefs = model_loss_coefs
        self.check_state = check_state
        self.token_dir = token_dir
        self.conf_dir = conf_dir
        self.save_stages = save_stages
        self.save_dir = save_dir

        self.dim = None
        self.max_depth = None
        self.model_size = None
        self.steps = None

    def initialize(self, 
                    use_split=True, 
                    split_rate_or_count=0.1,
                    use_shrink=False,
                    shrink_rate_or_count=0.5,
                    use_cache=True, 
                    use_aux_cache=True,
                    use_chat=False, 
                    chat_count=1,
                    use_acc=True,
                    model_dir=None,
                    eval_count=500
                    ):
        worker = self.worker

        worker.load_model(model_dir, token_dir=self.token_dir, dtype=worker.pred_dtype, distill=True)

        worker.load_sample(count=self.sample_count, ds='sft')
        if worker.aux_sample_dir is not None:
            worker.load_sample(count=None, ds='aux')

        if use_cache:
            worker.load_logit(self.sample_count, ds='sft')
        else:
            worker.calc_logit(ds='sft', bs=self.pbs)

        if worker.aux_logit_dir is not None:
            if use_aux_cache:
                worker.load_logit(None, ds='aux')
            else:
                worker.calc_logit(ds='aux', bs=self.pbs)

        if use_split:
            worker.split_sft_sample(split_rate_or_count, scatter=True)
        else:
            worker.aux_to_test()

        if use_shrink:
            worker.shrink_sft_sample(shrink_rate_or_count, scatter=True)

        worker.shuffle_sample_indices(ds='sft')
        if worker.aux_sample_indices:
            worker.shuffle_sample_indices(ds='aux')

        worker.model.to(worker.train_dtype)

        org_model_size = worker.count_params(head=False)
        steps = len(worker.sample_indices) // self.bs // self.itv
        org_max_depth = worker.n_layer
        self.max_depth = org_max_depth
        self.model_size = org_model_size
        self.steps = steps

        worker.log(
            f'sft_train:{len(worker.sample_indices)} sft_test:{len(worker.test_queries)} '
            f' bs:{self.bs} mbs:{self.mbs} r:{self.itv} sft_steps:{self.steps} model_size:{self.model_size / 1000 ** 3:.2f}')

        if use_chat:
            worker.star_log('org chat')
            embeddings = worker.test_embeddings[:chat_count] if worker.test_embeddings is not None else None
            worker.batch_chat(worker.test_queries[:chat_count], ans=worker.test_answers[:chat_count], embeddings=embeddings,
                              batch_size=self.pbs, max_log_count=1024, log_query=True, log_truth=True, log_answer=True,
                              log_space=True)

        if use_acc:
            worker.star_log('org acc')
            worker.calc_acc(count=eval_count, batch_size=self.pbs, max_log_count=worker.error_count, info='mode:org')

    def finetune(self, 
                 model_dir=None, 
                 use_split=True, 
                 split_rate_or_count=1000, 
                 use_shrink=False, 
                 shrink_rate_or_count=1.0, 
                 **kwargs):

        worker = self.worker
        worker.load_model(model_dir, token_dir=self.token_dir, dtype=worker.pred_dtype, distill=False)

        worker.load_sample(self.sample_count, ds='sft')
        if worker.aux_sample_dir is not None:
            worker.load_sample(None, ds='aux')

        if use_split:
            worker.split_sft_sample(split_rate_or_count, scatter=True)
        else:
            worker.aux_to_test()

        if use_shrink:
            worker.shrink_sft_sample(shrink_rate_or_count, scatter=True)

        worker.shuffle_sample_indices(ds='sft')
        if worker.aux_sample_indices:
            worker.shuffle_sample_indices(ds='aux')

        worker.model.to(worker.train_dtype)

        self.max_depth = worker.n_layer
        self.model_size = worker.count_params(head=False)
        self.steps = len(worker.sample_indices) // self.bs // self.itv

        worker.log(
            f'sft_train:{len(worker.sample_indices)} sft_test:{len(worker.test_queries)} '
            f' bs:{self.bs} mbs:{self.mbs} r:{self.itv} sft_steps:{self.steps} model_size:{self.model_size / 1000 ** 3:.2f}')

        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        model_steps = kwargs.get('model_steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        loss_coefs = kwargs.get('loss_coefs', {'emp': 1.0})
        train_stages = kwargs.get('train_stages', self.train_stages)
        save_stages = kwargs.get('save_stages', self.save_stages)

        for i in range(train_stages):
            ret = worker.train_model(bs=bs, mbs=mbs, steps=model_steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=None, mode='full', loss_coefs=loss_coefs, checkpoint=False,
                                     info=f'stage:sft_{i}', ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_model(save_dir=self.save_dir + f'_sft_{i}')
                worker.copy_conf(src_dir=self.conf_dir, dst_dir=self.save_dir + f'_sft_{i}')

    def load_ckpt(self, use_load=False, load_dir=None, eval_count=500):
        if not use_load:
            return
        worker = self.worker
        worker.star_log('reload ckpt')
        worker.load_ckpt(load_dir)
        worker.star_log('ckpt acc')
        worker.calc_acc(count=eval_count, batch_size=self.pbs, max_log_count=worker.error_count,
                             info=f'mode:reload layer:{worker.n_layer}')

    def depth_prune(self, use_depth=True, segs=None, **kwargs):
        if not use_depth:
            return
        worker = self.worker
        if segs is None:
            segs = range(self.max_depth - 1, 0, -1)
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        layer_steps = kwargs.get('layer_steps', self.steps)
        model_steps = kwargs.get('model_steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)
        checkpoint = kwargs.get('checkpoint', False)
        layer_indices = kwargs.get('layer_indices', None)

        for i in segs:
            if i >= worker.n_layer:
                worker.log(f'skip layer, index={i}')
                continue
            worker.star_log(f'depth_{i}')
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, steps=layer_steps, lr=lr, lrs=lrs, optim=optim, layer_indices=[i],
                               check_state=check_state, mode='depth', loss_coefs=layer_loss_coefs, info='stage:depth_' + str(i),
                               ds='sft')
            worker.clip_layer(i)
            if isinstance(layer_indices, float):
                layer_indices_ = range(int((1-layer_indices)*i),i)
            elif isinstance(layer_indices, int):
                layer_indices_ = range(i-layer_indices,i)
            else:
                layer_indices_ = layer_indices
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=model_steps, lr=lr, lrs=lrs, optim=optim, layer_indices=layer_indices_,
                                     check_state=check_state, mode='upper', loss_coefs=model_loss_coefs,
                                     checkpoint=checkpoint, info='stage:depth_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_depth_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_depth_{i}')
            if not ret['suc']:
                worker.log('stop depth pruning')
                break

    def mlp_prune(self, use_mlp=True, segs=None, **kwargs):
        if not use_mlp:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        layer_steps = kwargs.get('layer_steps', self.steps)
        model_steps = kwargs.get('model_steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)
        se = [int(round(x * self.worker.mlp_dim)) for x in segs]
        for i in range(len(segs)-1):
            worker.star_log(f'mlp_{i}')
            if se[i] <= 0 and se[i+1] <= 0:
                continue
            zero_counts = list(np.linspace(se[i], se[i+1], layer_steps, endpoint=True, dtype=np.int32))
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, zero_counts=zero_counts, lr=lr, lrs=lrs, optim=optim,
                               check_state=check_state, info='stage:mlp_' + str(i), mode='mlp',
                               loss_coefs=layer_loss_coefs, ds='sft')
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=model_steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=check_state, mode='upper', loss_coefs=model_loss_coefs,
                                     checkpoint=False, info='stage:mlp_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_mlp_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_mlp_{i}')
            if not ret['suc']:
                worker.log('stop mlp pruning')
                break

    def attn_prune(self, use_attn=True, segs=None, **kwargs):
        if not use_attn:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        layer_steps = kwargs.get('layer_steps', self.steps)
        model_steps = kwargs.get('model_steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        attn_mode = kwargs.get('attn_mode', 'attn')  # attn/head
        save_stages = kwargs.get('save_stages', self.save_stages)
        se = [int(round(x * self.worker.attn_dim)) for x in segs]
        for i in range(len(segs)-1):
            worker.star_log(f'attn_{i}')
            if se[i] <= 0 and se[i+1] <= 0:
                continue
            zero_counts = list(np.linspace(se[i], se[i+1], layer_steps, endpoint=True, dtype=np.int32))
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, zero_counts=zero_counts, lr=lr, lrs=lrs, optim=optim,
                               check_state=check_state, info='stage:attn_' + str(i), mode=attn_mode,
                               loss_coefs=layer_loss_coefs, ds='sft')
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=model_steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=check_state, mode='upper', loss_coefs=model_loss_coefs,
                                     checkpoint=False, info='stage:attn_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_attn_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_attn_{i}')
            if not ret['suc']:
                worker.log('stop attn pruning')
                break

    def dim_prune(self, use_dim=True, segs=None, **kwargs):
        if not use_dim:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        layer_steps = kwargs.get('layer_steps', self.steps)
        model_steps = kwargs.get('model_steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)
        se = [int(round(x * self.worker.dim)) for x in segs]
        for i in range(len(segs)-1):
            worker.star_log(f'dim_{i}')
            if se[i] <= 0 and se[i+1] <= 0:
                continue
            zero_counts = list(np.linspace(se[i], se[i+1], layer_steps, endpoint=True, dtype=np.int32))
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, zero_counts=zero_counts, lr=lr, lrs=lrs, optim=optim,
                               check_state=check_state, info='stage:dim_' + str(i), mode='dim',
                               loss_coefs=layer_loss_coefs, ds='sft')
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=model_steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=check_state, mode='upper', loss_coefs=model_loss_coefs,
                                     checkpoint=False, info='stage:dim_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_dim_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_dim_{i}')
            if not ret['suc']:
                worker.log('stop dim pruning')
                break

    def mlp_reparam(self, use_mlp_reparam=False, **kwargs):
        if not use_mlp_reparam:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        steps = kwargs.get('steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)

        worker.star_log('mlp reparam')
        for i in range(2):
            reparam_size = self.dim // 2
            worker.reparam(mode='mlp')
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, zero_counts=range(2 * self.dim - reparam_size, 2 * self.dim, self.itv),
                               lr=lr, lrs=lrs, optim=optim, info='stage:mlp_reparam_' + str(i), mode='mlp', loss_coefs=layer_loss_coefs,
                               ds='sft')
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=check_state, mode='block', loss_coefs=model_loss_coefs,
                                     info='stage:mlp_reparam_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_mlp_reparam_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_mlp_reparam_{i}')

    def attn_reparam(self, use_attn_reparam=False, **kwargs):
        if not use_attn_reparam:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        steps = kwargs.get('steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)

        worker.star_log('attn reparam')
        for i in range(2):
            reparam_size = self.dim // 16
            worker.reparam(mode='attn', size=reparam_size)
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, zero_counts=range(self.dim // 4 - reparam_size, self.dim // 4, self.itv),
                               lr=lr, lrs=lrs, optim=optim, info='stage:attn_reparam_' + str(i), mode='attn', loss_coefs=layer_loss_coefs,
                               ds='sft')
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=check_state, mode='block', loss_coefs=model_loss_coefs,
                                     info='stage:attn_reparam_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_attn_reparam_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_attn_reparam_{i}')

    def dim_reparam(self, use_dim_reparam=False, **kwargs):
        if not use_dim_reparam:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        steps = kwargs.get('steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        layer_loss_coefs = kwargs.get('layer_loss_coefs', self.layer_loss_coefs)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)

        worker.star_log('dim reparam')
        for i in range(2):
            reparam_size = self.dim // 16
            worker.reparam(mode='dim', size=reparam_size)
            worker.train_layer(bs=bs, mbs=mbs, pbs=pbs, zero_counts=range(self.dim // 4 - reparam_size, self.dim // 4, self.itv),
                               lr=lr, lrs=lrs, optim=optim, info='0', mode='stage:dim_reparam_' + str(i), loss_coefs=layer_loss_coefs,
                               ds='sft')
            ret = worker.train_model(bs=bs, mbs=mbs, pbs=pbs, steps=steps, lr=lr, lrs=lrs, optim=optim,
                                     check_state=check_state, mode='block', loss_coefs=model_loss_coefs,
                                     info='stage:dim_reparam_' + str(i), ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_dim_reparam_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_dim_reparam_{i}')

    def freeze_param(self, use_freeze=False, **kwargs):
        if not use_freeze:
            return
        worker = self.worker

        worker.star_log('eval state')
        worker.eval()

        worker.star_log('eval size')
        prune_model_size = worker.count_params(head=False)
        worker.log(
            f'org_size:{round(self.model_size / 1000 ** 3, 2)} prune_size:{round(prune_model_size / 1000 ** 3, 2)}')

        # worker.star_log('eval chat')
        # worker.batch_chat(worker.test_queries[:5],
        #                   embeddings=worker.test_embeddings[:5] if worker.test_embeddings is not None else None,
        #                   batch_size=self.pbs,
        #                   log_query=True,
        #                   log_answer=True)

        worker.star_log('eval acc')
        worker.calc_acc(count=None, batch_size=self.pbs, max_log_count=worker.error_count, info=f'mode:eval layer:{worker.n_layer}')

        save_stages = kwargs.get('save_stages', self.save_stages)
        if save_stages is not None and 0 in save_stages:
            worker.save_conf(save_dir=self.save_dir + f'_freeze')
            worker.save_ckpt(save_dir=self.save_dir + f'_freeze')

    def refit_param(self, use_refit=False, **kwargs):
        if not use_refit:
            return
        worker = self.worker
        bs = kwargs.get('bs', self.bs)
        mbs = kwargs.get('mbs', self.mbs)
        pbs = kwargs.get('pbs', self.pbs)
        model_steps = kwargs.get('model_steps', self.steps)
        lr = kwargs.get('lr', self.lr)
        lrs = kwargs.get('lrs', self.lrs)
        optim = kwargs.get('optim', self.optim)
        model_loss_coefs = kwargs.get('model_loss_coefs', self.model_loss_coefs)
        check_state = kwargs.get('check_state', self.check_state)
        save_stages = kwargs.get('save_stages', self.save_stages)
        train_stages = kwargs.get('train_stages', self.train_stages)

        worker.train()
        for i in range(train_stages):
            worker.star_log(f'refit_{i}')
            ret = worker.train_model(bs=bs, mbs=mbs, steps=model_steps, lr=lr, lrs=lrs, optim=optim,
                                    check_state=check_state, mode='upper', loss_coefs=model_loss_coefs, info=f'stage:refit_{i}',
                                    ds='sft')
            if save_stages is not None and i in save_stages:
                worker.save_conf(save_dir=self.save_dir + f'_refit_{i}')
                worker.save_ckpt(save_dir=self.save_dir + f'_refit_{i}')

    def final_export(self, use_final=False, **kwargs):
        if not use_final:
            return
        worker = self.worker
        worker.star_log('final save')
        worker.save_conf(save_dir=self.save_dir)
        worker.save_model(save_dir=self.save_dir)
        worker.star_log('final load')
        worker.load_model(self.save_dir, token_dir=self.token_dir, mask_dir=self.save_dir+'/mask.npz')
        worker.star_log('final chat')
        worker.batch_chat(worker.test_queries[:5],
                          ans=worker.test_answers[:5],
                          embeddings=worker.test_embeddings[:5] if worker.test_embeddings is not None else None,
                          batch_size=self.pbs, 
                          log_query=True, 
                          log_answer=True, 
                          log_truth=True, 
                          log_space=True)
        worker.star_log('final acc')
        worker.calc_acc(count=None, batch_size=self.pbs, max_log_count=worker.error_count, info=f'mode:final layer:{worker.n_layer}')

    def save(self, save_dir):
        self.worker.save_conf(save_dir=save_dir)
        self.worker.save_ckpt(save_dir=save_dir)