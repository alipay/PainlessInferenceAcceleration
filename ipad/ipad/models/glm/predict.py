# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


from __future__ import print_function

import time
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import sys
import copy
import types
import random
import warnings
import pandas as pd
import os 

import torch

from tokenization_glm import GLMChineseTokenizer
from modeling_glm import GLMForConditionalGeneration

# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

ts = time.time()
conf_dir = '/mntnlp/nanxiao/antglm_distill_prefetch_v0'
model_dir = '/mntnlp/nanxiao/antglm_distill_prefetch_v0'
tokenizer = GLMChineseTokenizer.from_pretrained(conf_dir)
print(f'load tokenizer in {round(time.time()-ts,3)}s')

dtype = torch.float16
model = GLMForConditionalGeneration.from_pretrained(model_dir
                                        ,cache_dir='/'
                                        ,torch_dtype=dtype
                                        ,low_cpu_mem_usage=True
                                        ,device_map='auto'
                                        )

print(f'load model in {round(time.time()-ts,3)}s')

# model.transformer.h = model.transformer.h[:1]
# model = model.half().cuda()
model.eval()

org_model = model
use_ds = True
if use_ds:
    import deepspeed
    # from deepspeed import DeepSpeedInferenceConfig
    # from deepspeed.inference.config import DeepSpeedTPConfig, QuantizationConfig
    # tp_conf = DeepSpeedTPConfig(enabled=False, tp_size=1)
    # quant_conf = QuantizationConfig(enabled=False)
    model = deepspeed.init_inference(model=model,
                                    dtype=torch.float16, 
                                    # enable_cuda_graph=True, 
                                    replace_with_kernel_inject=True, 
                                    # use_triton=True, 
                                    # triton_autotune=True,
                                    # triangular_masking=False, 
                                    # tensor_parallel=tp_conf,
                                    # quant=quant_conf
                                    )


def reload():
    importlib.reload(modeling_glm)
    patch()

def to_words(token_ids):
    if isinstance(token_ids, list):
        tokens = []
        for i in token_ids:
            tokens.append(tokenizer._convert_id_to_token(i))
        print(tokens)
    else:
        print(tokenizer._convert_id_to_token(token_ids))

def to_ids(tokens):
    return tokenizer._convert_token_to_id(tokens)

# /root/miniconda3/lib/python3.9/site-packages/transformers/generation/utils.py
# 1401:                 use_prefetch=use_prefetch,
# 1104:        use_prefetch: Optional[bool] = False,

def read_query(fix=True):
    filename = '/ossfs/workspace/lookahead/dataset/search_queries_6k.txt'
    qs = []
    for line in open(filename,'r'):
        q = line.strip()
        qs.append(q)
    return qs


def chat(query, max_length=256):
    if isinstance(query, list):
        inputs = tokenizer(query, 
                        padding=True, 
                        truncation=False, 
                        return_tensors="pt")
    else:
        inputs=tokenizer(query, 
                        return_tensors="pt", 
                        padding='max_length', 
                        truncation=True
                        )
    input_length = inputs['input_ids'].shape[1]
    # print(f'input_length:{input_length} max_length:{max_length}', [(x,y.shape) for x,y in inputs.items()])
    inputs=tokenizer.build_inputs_for_generation(inputs,
                                                 max_gen_length=max_length
                                                 )
    # print(f'build:', [(x,y.shape) for x,y in inputs.items()])
    inputs = {k: -10000*(1.0-v.to(dtype=dtype).cuda()) if k=='generation_attention_mask' else v.cuda() for k,v in inputs.items()}
    outputs = model.generate(input_ids=inputs['input_ids'],
                             attention_mask=inputs['generation_attention_mask'],
                             position_ids=inputs['position_ids'],
                             eos_token_id=tokenizer.eop_token_id,
                             use_cache=True,
                             max_length=max_length+input_length,
                             repetition_penalty=1.0,
                             do_sample=False,
                            #  top_k=10,
                            #  num_beams=1,
                            #  top_p=0.9,
                             temperature=1.0,
                             # num_beams=3,
                             # num_return_sequences=1,
                             pad_token_id=tokenizer.eos_token_id,
                            #  bos_token_id=tokenizer.bos_token_id,
                            #  eos_token_id=tokenizer.eos_token_id,
                            #  eos_token_id=tokenizer.eop_token_id
                             )
    outputs = outputs[:, input_length+1:]
    results = []
    token_list = []
    for output_id in outputs:
        token_ids = output_id.tolist()
        if 50005 in token_ids:
            token_ids = token_ids[:token_ids.index(50005)+1]
        token_list.append(token_ids)
        responce_txt = tokenizer.decode(token_ids)
        responce_txt = responce_txt.replace("<|startofpiece|>", " ").replace("<|endofpiece|>"," ")
        results.append(responce_txt)
    return results, inputs['input_ids'], token_list


def batch_chat(qs, max_length=256, batch_size=1):
    n_repeat = 1
    total_out_tokens = 0
    total_times = 0
    chat_count = len(qs)
    for i in range(chat_count//batch_size):
        query = qs[i*batch_size:(i+1)*batch_size]
        speeds = []
        in_char = 0
        in_token = 0 
        out_char = 0
        out_token = 0
        ts = time.time()
        for k in range(n_repeat):
            results, input_tensors, output_tensor_ids = chat(query, 
                                                                max_length=max_length)
            in_char += sum([len(x) for x in query])
            in_token += input_tensors.shape[1]*batch_size
            out_char += sum([len(x) for x in results])
            out_token += sum([len(x) for x in output_tensor_ids])
        in_char /= n_repeat
        in_token /= n_repeat
        out_char /= n_repeat
        out_token /= n_repeat
        t = (time.time() - ts)/n_repeat
        speed_char = out_char/t
        speed_token = out_token/t
        speeds.append(speed_token)
        total_out_tokens += out_token
        total_times += t
        # print(f"Human:{query[:80]}...")
        print(f"Robot:{ '->->->'.join(results)}")
        print(f"{i}/{chat_count} input:{round(in_char,1)}/{round(in_token,1)} output:{round(out_char,1)}/{round(out_token,1)} time:{round(t,3)} speed:{round(speed_char,1)}/{round(speed_token,1)}\n")
    print(f'speed:{round(total_out_tokens/total_times,2)}')


# CUDA_VISIBLE_DEVICES=0 python -i antglm_prefetch_check.py

def to_profile(qs):
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    for q in qs:
        chat(q)
    pr.disable()
    s = io.StringIO()
    # sortby = SortKey.CUMULATIVE SortKey.TIME  
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME).print_stats(64)
    print(s.getvalue())

qs = read_query(fix=False)
chat(qs[0:2])[0]


batch_chat(qs[:10], batch_size=1)

# f = open('perf_check.log','w+')
# chat_count = 1000
# for warmup_count in [0,1000,3000,5000]:
#    batch_chat(qs[:10], prefetch_size=63, prefetch_length=8, debug_prefetch=True, erase=True, batch_size=2)
# f.close()

print('done!')
