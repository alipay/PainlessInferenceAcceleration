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

from lookahead.models.qwen.modeling_qwen import QWenLMHeadModel
from lookahead.models.qwen.tokenization_qwen import QWenTokenizer
from local_path import local_path_dict


model_dir = local_path_dict.get('qwen_quant', 'your/model/path') 

# should install auto_gptq at first
model = QWenLMHeadModel.from_pretrained(model_dir,
                                        device_map="auto",
                                        trust_remote_code=True
                                       )
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

tokenizer = QWenTokenizer.from_pretrained(model_dir)
stop_words = [tokenizer.encode(x)[0] for x in [',', '.', ' ', '，','。']]

# prompt = "杭州在哪里？"
prompt = "编一个200字左右的儿童故事"

for use_lookahead in [False, False, True, True]:
    debug_lookahead = False
    decoding_length = 64
    branch_length = 12
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length,
                       "stop_words": stop_words}
    model.generation_config.decoding_kwargs=decoding_kwargs
    model.generation_config.do_sample=False  # default is True for qwen, result in different responses in every generation
    ts = time.time()
    response, history = model.chat(tokenizer, prompt, history=None, eos_token_id=151645)
    te = time.time()
    token_count = len(tokenizer.encode(response))
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:\n{response}\n')
