# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""



import time 
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from lookahead.models.mixtral.modeling_mixtral import MixtralForCausalLM
from local_path import local_path_dict

# only worker with transformers>=4.36.0

model_dir = local_path_dict.get('mixtral', 'your/model/path') 
tokenizer = AutoTokenizer.from_pretrained(model_dir)


# note: this model cannot be fully loaded into a A100, refer to mixtral_quant_example.py for solution
model = MixtralForCausalLM.from_pretrained(model_dir
                                            , cache_dir='/'
                                            , torch_dtype=torch.float16
                                            , low_cpu_mem_usage=True
                                            , device_map={"":"cuda:0"})

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
attention_mask = inputs.attention_mask.cuda()
position_ids = None

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
    input_text = tokenizer.decode(input_ids[0])
    te = time.time()
    token_count = len(output_ids)
    print(f'lookahead:{use_lookahead} time:{te - ts:.3f}s speed:{token_count/(te-ts):.1f}token/s response:\n{response}\n')


