# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer

from lookahead.models.llama.modeling_llama import LlamaForCausalLM
from local_path import local_path_dict

model_dir = local_path_dict.get('llama', 'your/model/path') 

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = LlamaForCausalLM.from_pretrained(model_dir
                                         , cache_dir='../'
                                         , torch_dtype=dtype
                                         , low_cpu_mem_usage=True
                                         , device_map={"":"cuda:0"}
                                         )
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
stop_words = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
attention_mask = inputs.attention_mask.cuda()
position_ids = None

# first time without lookahead
for use_lookahead in [False, False, True, True]:
    debug_lookahead = False
    decoding_length = 64
    branch_length = 12
    ts = time.time()
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_length": decoding_length,
                       "branch_length": branch_length,
                       "stop_words": stop_words}
    streamer = TextIteratorStreamer(tokenizer)
    generator = model.stream_generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             streamer=streamer,
                             decoding_kwargs=decoding_kwargs
                             )

    response = ''
    for word in generator:
        response = response + word
        print(word, end='',flush=True)
    te = time.time()
    token_count = len(tokenizer.encode(response))-input_ids.size(1)
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s\n')

