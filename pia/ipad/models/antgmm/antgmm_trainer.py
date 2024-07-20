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
from rouge_score import rouge_scorer

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.autograd import Variable
from transformers.activations import gelu
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models.glm.tokenization_glm import GLMChineseTokenizer
from models.glm.modeling_glm import GLMForConditionalGeneration
from models.glm.glm_trainer import DistillPipe, GlmDistillWorker


class AntgmmDistillWorker(GlmDistillWorker):
    def __init__(self,
                 sample_dir=None, logit_dir=None, emb_dir=None, emb_idx=1,
                 aux_sample_dir=None, aux_logit_dir=None, aux_emb_dir=None,
                 log_dir=None, log_steps=1, eval_steps=5, eval_count=100,
                 max_input_len=256, max_gen_len=64,
                 train_dtype=torch.bfloat16,
                 pred_dtype=torch.float16,
                 device=None):
        super(AntgmmDistillWorker, self).__init__(log_dir=log_dir)
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

    def calc_acc(self, count=50000, batch_size=16, ds='test', max_log_count=0, filename=None, info=''):
        assert ds in ('aux','test')
        ts = time.time()
        queries = self.test_queries if ds == 'test' else self.aux_queries
        answers = self.test_answers if ds == 'test' else self.aux_answers
        embeddings = self.test_embeddings if ds == 'test' else self.aux_embeddings

        rs = self.batch_chat(queries[:count],
                              embeddings=embeddings[:count],
                              batch_size=batch_size,
                              max_length=32,
                              log_query=False,
                              log_truth=False,
                              log_answer=False,
                              log_space=False,
                              emit=True)
        te = time.time()
        n_p = 0
        n = 0
        errors = []
        n_ps = [0]*((len(rs)-1)//100+1)
        for i, pred in enumerate(rs):
            pred, gt = self.normalize(queries[i], pred, answers[i])
            n+=1
            if pred == gt:
                n_p += 1
                n_ps[i//100] += 1
            else:
                errors.append([i, queries[i].replace('[CLS]',''),  gt,  pred])
        acc = n_p/n
        size = self.est_params()/1e9
        self.log(f'\naccuracy:{acc:.3f} size:{size:.3f} sample:{n} correct:{n_p} details:{n_ps} elapse:{te-ts:.3f} {info}\n')
        for i, q,t,p in errors[:max_log_count]:
            self.log(f'index:{i} true:{t} pred:{p} query:{q}')
        return acc

    def normalize(self, prompt, pred, gt):
        """ '行业': '结合标题描述图片商品的行业类别，标题为',
            '主体': 'Observe the image carefully and describe the main body of the product',
            '背景颜色': '描述图片商品的背景颜色',
            '表现形式': '描述图片商品的表现形式',
            '牛皮癣': '描述图片商品的牛皮癣程度',
            'LOGO': '仔细观察图片的局部区域，告诉我图中是否存在商品LOGO',
        """
        pred = pred.strip()
        if '描述图片商品的行业类别' in prompt:
            pred = pred.replace('图片商品的行业类别是:','').replace('图片商品行业类别是:','').replace('图片商品的行业类别号是:','').replace('<|endofpiece|>','')
            gt = gt.replace('图片商品的行业类别是:','').replace('图片商品行业类别是:','').replace('图片商品的行业类别号是:','')
        elif 'describe the main body of the product' in prompt:
            if pred in ['The image does not have a main product.', '图片商品没有主体']:
                pred = 'non'
            else:
                pred = pred.replace('the main product of the image is:','')
        elif '图中是否存在商品LOGO' in prompt:
            logo_dict = {"是的，图片中有LOGO": "是", "不，图片中没有LOGO":"否", "是的,图片中有LOGO":"是", "不,图片中没有LOGO":"否"}
            if pred in logo_dict:
                pred = logo_dict[pred]
        elif '描述图片商品的背景颜色' in prompt:
            if pred == '图片商品没有背景':
                pred = 'non'
            else:
                pred = pred.replace('图片商品的背景颜色是:','')
        elif '描述图片商品的表现形式' in prompt:
            pred = pred.replace('图片商品的表现形式是:','')
        elif '描述图片商品的牛皮癣程度' in prompt:
            pred = pred.replace('图片商品的牛皮癣程度是:','')
        else:
            raise ValueError(F'UNKNOWN PROMPT:{prompt}')
        return pred.strip(), gt.strip()

