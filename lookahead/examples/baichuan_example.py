# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import sys
import time

import torch
from transformers import AutoTokenizer


sys.path.append('..')
from common.lookahead_cache import LookaheadCache
from models.baichuan.modeling_baichuan import BaichuanForCausalLM
from models.baichuan.tokenization_baichuan import BaichuanTokenizer
from transformers.generation.utils import GenerationConfig

model_dir = 'your/model/path'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = BaichuanForCausalLM.from_pretrained(model_dir
                                            , cache_dir='../'
                                            , torch_dtype=torch.float16
                                            , low_cpu_mem_usage=True
                                            , device_map='auto'
                                            )
tokenizer = BaichuanTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
stop_ids = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))
lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
model.lookahead_cache = lookahead_cache

prompt = "Hello, I'm am conscious and"

# first time without lookahead
for use_lookahead in [False, False, True, True]:
    debug_lookahead = False
    decoding_length = 64
    branch_length = 12
    ts = time.time()
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length}

    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.generation_config.decoding_kwargs = decoding_kwargs
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.do_sample = False

    messages = []
    messages.append({"role": "user", "content": prompt})

    response = model.chat(tokenizer, messages)
    te = time.time()
    if use_lookahead:
        print(f'with lookahead:{te - ts:.3f}s')
    else:
        print(f'without lookahead:{te - ts:.3f}s')
    print(f'prompt:{prompt}')
    print(f'input text:{prompt}')
    print(f'output text:{response}')
