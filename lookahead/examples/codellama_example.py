# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import os
import sys
import time

import torch

from lookahead.models.llama.modeling_llama import LlamaForCausalLM
from local_path import local_path_dict
from transformers import CodeLlamaTokenizer

model_dir = local_path_dict.get('codellama', 'your/model/path') 

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# dtype = torch.float32
model = LlamaForCausalLM.from_pretrained(model_dir
                                         , cache_dir='../'
                                         , torch_dtype=dtype
                                         , low_cpu_mem_usage=True
                                         , device_map={"":"cuda:0"}
                                         )
tokenizer = CodeLlamaTokenizer.from_pretrained(model_dir)


prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''

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
    max_new_tokens = 128
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
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
                             decoding_kwargs=decoding_kwargs
                             )
    output_ids = outputs
    input_length = input_ids.size(-1)
    output_ids = output_ids[0, input_length:].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    te = time.time()
    # Here we just follow the official examples, you can define your own postprocess.
    response = prompt.replace("<FILL_ME>", response)
    token_count = len(output_ids)
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:{response}\n\n\n')

