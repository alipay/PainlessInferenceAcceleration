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

from pia.lookahead.common.pretrained_model import LookaheadCache
from pia.lookahead.models.qwen.modeling_qwen import QWenLMHeadModel
from pia.lookahead.models.qwen.tokenization_qwen import QWenTokenizer
from pia.lookahead.examples import local_path_dict


model_dir = local_path_dict.get('qwen_quant', 'your/model/path') 

# test with transformers==4.36.0
model = QWenLMHeadModel.from_pretrained(model_dir,
                                        device_map="auto",
                                        trust_remote_code=True
                                       )

tokenizer = QWenTokenizer.from_pretrained(model_dir)
stop_words = [tokenizer.encode(x)[0] for x in [',', '.', ' ', '，','。']]

# prompt = "杭州在哪里？"
prompt = "编一个200字左右的儿童故事"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    model.generation_config.do_sample=False
    model.generation_config.repetition_penalty=None  # repetition_penalty is not fully supported currently, will fix in the future
    ts = time.time()
    response, history = model.chat(tokenizer, prompt, history=None, eos_token_id=151645)
    te = time.time()
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{len(response)/(te-ts):.1f}c/s response:{response}\n\n\n')
