# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from pia.lookahead.models.internlm.modeling_internlm2 import InternLM2ForCausalLM
from pia.lookahead.models.internlm.tokenization_internlm import InternLMTokenizer   
from pia.lookahead.examples import local_path_dict

model_dir = local_path_dict.get('internlm', 'your/model/path') 
# model_dir = '/mntnlp/liangchen/internlm2-chat-7b'

tokenizer = InternLMTokenizer.from_pretrained(model_dir)
model = InternLM2ForCausalLM.from_pretrained(model_dir
                                         , cache_dir='../'
                                         , torch_dtype=torch.float16
                                         , low_cpu_mem_usage=True
                                         , device_map={"":"cuda:0"}
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
stop_words = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))

prompt = "做一个自我介绍"

# first time without lookahead
for use_lookahead in [True, True]:
    debug_lookahead = True
    decoding_length = 64
    branch_length = 12
    ts = time.time()
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_length": decoding_length,
                       "branch_length": branch_length,
                       "stop_words": stop_words,
                       "tokenizer": tokenizer}
    # model.generation_config.decoding_kwargs=decoding_kwargs
    ts = time.time()
    response, history = model.chat(tokenizer, prompt, history=[], decoding_kwargs=decoding_kwargs)
    te = time.time()
    token_count = len(tokenizer.encode(response))
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:\n{response}\n')
    # output_ids = outputs
    # input_length = input_ids.size(-1)
    # output_ids = output_ids[0, input_length:].tolist()
    # response = tokenizer.decode(output_ids)
    # input_text = tokenizer.decode(input_ids[0])
    # te = time.time()
    # token_count = len(output_ids)
    # print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:{response}\n\n\n')

