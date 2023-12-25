# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import sys
import time
import torch

# sys.path.append('..')
from pia.common.pretrained_model import LookaheadCache
from pia.models.glm.modeling_glm import GLMForConditionalGeneration
from pia.models.glm.tokenization_glm import GLMChineseTokenizer


model_dir = 'your/model/path'
model = GLMForConditionalGeneration.from_pretrained(model_dir
                                                    , cache_dir='../'
                                                    , offload_folder='./'
                                                    , torch_dtype=torch.float16
                                                    , low_cpu_mem_usage=True
                                                    , device_map='auto')
tokenizer = GLMChineseTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
lookahead_cache = LookaheadCache(eos=50005, stop_words={43359, 43360, 43361, 43362})
model.lookahead_cache = lookahead_cache

for block in model.glm.transformer.layers:
    block.attention._norm()

# prompt = "Hello, I'm am conscious and"
prompt = "杭州在哪里？[gMASK]"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False, )

device = model.device
debug_lookahead = True
decoding_length = 64
branch_length = 12
max_new_tokens = 128
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=max_new_tokens + decoding_length + 3)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['generation_attention_mask'].to(device)
position_ids = inputs['position_ids'].to(device)

for use_lookahead in [False,False,True,True]:
    ts = time.time()
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length}
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eop_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             decoding_kwargs=decoding_kwargs
                             )
    output_ids = outputs
    input_length = input_ids.size(-1)
    output_ids = output_ids[:, input_length:].tolist()
    # output_ids = output_ids.tolist()
    output_texts = []
    output_id_list = []
    for token_ids in output_ids:
        output_id_list.append(token_ids)
        text = tokenizer.decode(token_ids)
        output_texts.append(text)
    input_id_list = input_ids.tolist()
    te = time.time()
    print(f'use_lookahead:{use_lookahead} time:{te - ts:.3f} output:{output_texts}')
