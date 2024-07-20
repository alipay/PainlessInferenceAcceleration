# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


from __future__ import print_function

import time
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import copy
import types
import random
import warnings
import pandas as pd
import os 

import numpy as np

import torch

import sys
sys.path.append('.')
from tokenization_glm import GLMChineseTokenizer
import modeling_glm
from modeling_glm import GLMForConditionalGeneration

# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

ts = time.time()
conf_dir = '/mntnlp/nanxiao/blip'
model_dir = '/mntnlp/nanxiao/blip'
tokenizer = GLMChineseTokenizer.from_pretrained(conf_dir)
print(f'load tokenizer in {round(time.time()-ts,3)}s')

dtype = torch.float16
model = GLMForConditionalGeneration.from_pretrained(model_dir
                                        ,cache_dir='./'
                                        ,torch_dtype=dtype
                                        ,low_cpu_mem_usage=True
                                        ,device_map='auto'
                                        )

sd = torch.load(model_dir+'/pytorch_model.bin', map_location=torch.device('cpu'))
input_emb_mask = sd.get('input_emb_mask', None)

print(f'load model in {round(time.time()-ts,3)}s')

# model.transformer.h = model.transformer.h[:1]
# model = model.half().cuda()
model.eval()

def read_query():
    filename = '/mntnlp/nanxiao/blip/qas_10k.txt'
    qs = []
    ans = []
    for line in open(filename,'r'):
        q,a = line.strip().split('[gMASK]')
        qs.append(q+'[gMASK]')
        ans.append(a)
    filename = '/mntnlp/nanxiao/blip/embs_10k.bin'
    with open(filename, 'rb') as f:
        caches = np.load(f)
    if input_emb_mask is not None:
        caches = caches[:,:,input_emb_mask.cpu().numpy()==1.0]
    return qs,ans, caches


def chat(query, embs=None, max_length=256):
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
    input_ids = inputs['input_ids'].cuda()
    embs = torch.from_numpy(embs).to(dtype).cuda()

    # labels = inputs['labels'].cuda()
    position_ids = inputs['position_ids'].cuda()
    attention_mask = -10000.0*(1.0-inputs['generation_attention_mask'].to(dtype).cuda())
    
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             inputs_embeds=embs,
                             eos_token_id=tokenizer.eop_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_length=max_length+input_length,
                             repetition_penalty=1.0,
                             do_sample=False,
                            #  top_k=10,
                            #  num_beams=1,
                            #  top_p=0.9,
                            #  temperature=1.0,
                            #  num_beams=3,
                            #  num_return_sequences=1,
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


def batch_chat(qs, embeddings=None, max_length=256, batch_size=1):
    n_repeat = 1
    total_out_tokens = 0
    total_times = 0
    chat_count = len(qs)
    for i in range(chat_count//batch_size):
        query = qs[i*batch_size:(i+1)*batch_size]
        embs = embeddings[i*batch_size:(i+1)*batch_size]
        speeds = []
        in_char = 0
        in_token = 0 
        out_char = 0
        out_token = 0
        ts = time.time()
        for k in range(n_repeat):
            results, input_tensors, output_tensor_ids = chat(query, 
                                                            embs=embs,
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
        print(f"Human:{query}")
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

qs,ans, caches = read_query()
chat(qs[0:2], embs=caches[0:2])[0]

s = 9239
e = s + 10
batch_chat(qs[s:e], embeddings=caches[s:e], batch_size=1, max_length=128)

print('done!')
