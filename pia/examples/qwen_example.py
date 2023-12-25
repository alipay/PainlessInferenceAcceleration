# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import os
import sys
import time
import torch
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig

# sys.path.append('..')
from pia.common.pretrained_model import LookaheadCache
from pia.models.qwen.modeling_qwen import QWenLMHeadModel
from pia.models.qwen.tokenization_qwen import QWenTokenizer

model_dir = 'your/model/path'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = QWenLMHeadModel.from_pretrained(model_dir
                                       , cache_dir='../'
                                       , torch_dtype=dtype
                                       , low_cpu_mem_usage=True
                                       , device_map='auto'
                                       ).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

tokenizer = QWenTokenizer.from_pretrained(model_dir)
stop_ids = set(tokenizer.convert_tokens_to_ids([',', '.', ' ', '，','。']))
lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
model.lookahead_cache = lookahead_cache


prompt = "杭州在哪里？"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

for use_lookahead in [False, False, True]:
    debug_lookahead = False
    decoding_length = 64
    branch_length = 12
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length}
    model.generation_config.decoding_kwargs = decoding_kwargs
    ts = time.time()
    response, history = model.chat(tokenizer, prompt, history=None)
    te = time.time()
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s response:{response}')
