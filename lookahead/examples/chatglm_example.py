# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import sys
import time
import torch

from lookahead.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from lookahead.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from local_path import local_path_dict

model_dir = local_path_dict.get('chatglm', 'your/model/path') 

tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
model = ChatGLMForConditionalGeneration.from_pretrained(model_dir
                                                                , cache_dir='/'
                                                                , torch_dtype=torch.float16
                                                                , low_cpu_mem_usage=True
                                                                , device_map={"":"cuda:0"}
                                                                )
stop_words = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))

# prompt = "Hello, I'm am conscious and"
prompt = "杭州在哪里？"

inputs = model.build_inputs(tokenizer, prompt, history=[])
input_ids = inputs.input_ids.cuda()
attention_mask = inputs.attention_mask.cuda()
position_ids = None

device = model.device
debug_lookahead = False
decoding_length = 64
branch_length = 12
max_new_tokens = 128


for use_lookahead in [False,False,True,True]:
    ts = time.time()
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_length": decoding_length,
                       "branch_length": branch_length,
                       "stop_words": stop_words}
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             decoding_kwargs=decoding_kwargs
                             )
    output_ids = outputs
    input_length = input_ids.size(-1)
    output_ids = output_ids[0, input_length:].tolist()
    response = tokenizer.decode(output_ids)
    te = time.time()
    token_count = len(output_ids)
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:\n{response}\n')

