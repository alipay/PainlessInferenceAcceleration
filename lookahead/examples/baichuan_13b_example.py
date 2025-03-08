# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig


from lookahead.models.baichuan_13b.modeling_baichuan import BaichuanForCausalLM
from lookahead.models.baichuan_13b.tokenization_baichuan import BaichuanTokenizer
from local_path import local_path_dict

model_dir = local_path_dict.get('baichuan_13b', 'your/model/path') 


tokenizer = BaichuanTokenizer.from_pretrained(model_dir)
model = BaichuanForCausalLM.from_pretrained(model_dir
                                            , cache_dir='../'
                                            , torch_dtype=torch.float32
                                            , low_cpu_mem_usage=True
                                            , device_map={"": "cuda:0"}
                                            )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
stop_words = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))

prompt = "Hello, I'm am conscious and"

# first time without lookahead
for use_lookahead in [False, False, True, True]:
    debug_lookahead = False
    decoding_length = 64
    branch_length = 4
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_length": decoding_length,
                       "branch_length": branch_length,
                       "stop_words": stop_words,
                       "tokenizer": tokenizer}

    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    model.generation_config.decoding_kwargs = decoding_kwargs
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.do_sample = False
    model.generation_config.top_k = 50
    model.generation_config.top_p = 1.0
    model.generation_config.temperature = 1.0
    model.generation_config.repetition_penalty = 1.0

    messages = []
    messages.append({"role": "user", "content": prompt})

    ts = time.time()
    response = model.chat(tokenizer, messages)
    te = time.time()
    # print(f'prompt:{prompt}')
    # print(f'input text:{prompt}')
    token_count = len(tokenizer.encode(response))
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:\n{response}\n')
