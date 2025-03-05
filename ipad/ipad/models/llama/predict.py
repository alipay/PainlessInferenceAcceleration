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

import torch

import sys
from transformers import LlamaForCausalLM, LlamaTokenizer


# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

ts = time.time()
# conf_dir = '/mntnlp/nanxiao/llama'
# model_dir = '/mntnlp/nanxiao/llama'

conf_dir = '/mntnlp/common_base_model/llama_13b'
model_dir = '/mntnlp/common_base_model/llama_13b'

# conf_dir = '/mntnlp/common_base_model/vicuna_7b'
# model_dir = '/mntnlp/common_base_model/vicuna_7b'

tokenizer = LlamaTokenizer.from_pretrained(conf_dir)
print(f'load tokenizer in {round(time.time()-ts,3)}s')

# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'
prompt = 'hello yes'
print(tokenizer(prompt, max_length=8, padding='max_length', truncation=True))
# dtype = torch.float16
# model = LlamaForCausalLM.from_pretrained(model_dir
#                                         ,cache_dir='./'
#                                         ,torch_dtype=dtype
#                                         ,low_cpu_mem_usage=True
#                                         ,device_map='auto'
#                                         )

print(f'load model in {round(time.time()-ts,3)}s')

# model = model.half().cuda()

# model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
# model.config.bos_token_id = 1
# model.config.eos_token_id = 2

# model.eval()


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

def read_query(filename):
    qs = []
    for line in open(filename,'r'):
        q = line.strip().split('\t\t\t\t')[0]
        qs.append(q)

    return qs


def chat(query, max_length=256):
    inputs = tokenizer(query, return_tensors="pt")
    # print([(k,v,) for k,v in inputs.items()])
    input_length = inputs["input_ids"].size(-1)
    inputs = {k:v.cuda() for k,v in inputs.items()}
    outputs = model.generate(**inputs,
                            #  eos_token_id=tokenizer.eop_token_id,
                             use_cache=True,
                             max_length=max_length+input_length,
                            #  repetition_penalty=1.0,
                            #  do_sample=False,
                            #  top_k=10,
                            #  num_beams=1,
                            #  top_p=0.9,
                            #  temperature=1.0,
                             # num_beams=3,
                             # num_return_sequences=1,
                            #  pad_token_id=tokenizer.eos_token_id,
                            #  bos_token_id=tokenizer.bos_token_id,
                            #  eos_token_id=tokenizer.eos_token_id,
                            #  eos_token_id=tokenizer.eop_token_id
                             )
    outputs = outputs[:, input_length:]
    results = []
    token_list = []
    for output_id in outputs:
        token_ids = output_id.tolist()
        token_list.append(token_ids)
        responce_txt = tokenizer.decode(token_ids)
        # responce_txt = responce_txt.replace("<|startofpiece|>", " ").replace("<|endofpiece|>"," ")
        results.append(responce_txt)
    return query, inputs['input_ids'], results, token_list


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
        for k in range(batch_size):
            print(f"Human:{query[k]}")
            print(f"Robot:{results[k]}")
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

# prompt = 'The universe is the entirety of space, time, matter, and energy that exists.'
# prompt = '10 steps to build an ios app:'
# prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction:When did Virgin Australia start operating?### Input:Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.### Response:"
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.### Instruction:Which is a species of fish? Tope or Rope### Response:"

input_text,input_tokens, output_text, output_tokens= chat(prompt)
print(output_text)



inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), max_length=512, no_repeat_ngram_size=2)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

print('done!')
