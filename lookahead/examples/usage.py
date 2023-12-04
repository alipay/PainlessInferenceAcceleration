# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import time
from operator import itemgetter
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import copy
import types
import json
import random
import warnings
import pandas as pd
import os
import sys
import cProfile, pstats, io
from pstats import SortKey
import torch

sys.path.append('../../lookahead')
from common.pretrained_model import PrefetchCache

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import AutoTokenizer

from models.modeling_llama_fast import LlamaForCausalLM

model_dir = '/mntnlp/common_base_model/llama2-7b-chat'
model = LlamaForCausalLM.from_pretrained(model_dir
                                                , cache_dir='./'
                                                , torch_dtype=torch.float16
                                                , low_cpu_mem_usage=True
                                                , device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
prefetch_cache = PrefetchCache(eos=tokenizer.eos_token_id, stop_words={1919, 869, 259, 1577})
model.prefetch_cache = prefetch_cache

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
attention_mask = inputs.attention_mask.cuda()
position_ids = None

# first time without lookahead
for use_prefetch in [False, False, True, True]:
    debug_prefetch = False
    prefetch_size = 63
    prefetch_length = 12
    ts = time.time()
    outputs = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                                max_new_tokens=256,
                                repetition_penalty=1.0,
                                do_sample=False,
                                use_prefetch=use_prefetch,
                                debug_prefetch=debug_prefetch,
                                prefetch_size=prefetch_size,
                                prefetch_length=prefetch_length,
                                return_dict_in_generate=True
                                )
    output_ids = outputs.sequences
    input_length = input_ids.size(-1)
    output_ids = output_ids[0, input_length:].tolist()
    output_text = tokenizer.decode(output_ids)
    input_text = tokenizer.batch_decode(input_ids[0])
    te = time.time()
    if use_prefetch:
        print(f'with lookahead:{te-ts:.3f}s')
    else:
        print(f'without lookahead:{te-ts:.3f}s')
    print(f'prompt:{prompt}')
    print(f'input text:{input_text}')
    print(f'output text:{output_text}')

