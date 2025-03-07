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

from lookahead.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from local_path import local_path_dict

model_dir = local_path_dict.get('qwen2', 'your/model/path')

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Qwen2ForCausalLM.from_pretrained(model_dir
                                       , cache_dir='../'
                                       , torch_dtype="auto"
                                       , low_cpu_mem_usage=True
                                       , device_map={"": device}
                                       )

tokenizer = AutoTokenizer.from_pretrained(model_dir)
# stop_words = [tokenizer.encode(x)[0] for x in [',', '.', ' ', '，','。']]
prompt = "杭州在哪里？"
inputs = tokenizer(prompt, return_tensors="pt")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)


for use_lookahead in [False, False, True, True]:
    decoding_length = 64
    branch_length = 12
    debug_lookahead = False
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_length": decoding_length,
                       "branch_length": branch_length
                       }

    ts = time.time()
    outputs = model.generate(input_ids=model_inputs.input_ids,
                             attention_mask=model_inputs.attention_mask,
                             position_ids=None,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             decoding_kwargs=decoding_kwargs
                             )
    output_ids = outputs
    input_length = model_inputs.input_ids.size(-1)
    output_ids = output_ids[0, input_length:].tolist()
    response = tokenizer.decode(output_ids)
    # input_text = tokenizer.decode(input_ids[0])
    te = time.time()
    token_count = len(tokenizer.encode(response))
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:\n{response}\n')
