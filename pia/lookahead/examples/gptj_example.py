# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import os
import sys
import time
import torch
from transformers import AutoTokenizer

from pia.lookahead.common.pretrained_model import LookaheadCache
from pia.lookahead.models.gptj.modeling_gptj import GPTJForCausalLM
from pia.lookahead.examples import local_path_dict

model_dir = local_path_dict.get('gptj', 'your/model/path') 

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = GPTJForCausalLM.from_pretrained(model_dir
                                       , cache_dir='../'
                                       , torch_dtype=dtype
                                       , low_cpu_mem_usage=True
                                       , device_map='auto'
                                       )
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
stop_ids = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))
lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
model.lookahead_cache = lookahead_cache

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)
position_ids = None

for use_lookahead in [False, False, True, True]:
    debug_lookahead = True
    decoding_length = 64
    branch_length = 12
    ts = time.time()
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length}
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             decoding_kwargs=decoding_kwargs,
                             )
    output_ids = outputs
    input_length = input_ids.size(-1)
    output_ids = output_ids[0, input_length:].tolist()
    output_text = tokenizer.decode(output_ids)
    input_text = tokenizer.decode(input_ids[0])
    te = time.time()
    if use_lookahead:
        print(f'with lookahead:{te - ts:.3f}s')
    else:
        print(f'without lookahead:{te - ts:.3f}s')
    print(f'prompt:{prompt}')
    print(f'input text:{input_text}')
    print(f'output text:{output_text}')
