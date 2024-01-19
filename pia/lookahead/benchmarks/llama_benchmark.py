# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import sys
import torch
from transformers import AutoTokenizer

from pia.lookahead.common.lookahead_cache import LookaheadCache
from benchmark import Benchmark


class LlameBenchmark(Benchmark):

    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        # org version for llama
        # from pia.lookahead.models.llama.modeling_llama import LlamaForCausalLM
        # fused op version for llama
        from pia.lookahead.models.llama.modeling_llama_batch import LlamaForCausalLM 
        model = LlamaForCausalLM.from_pretrained(model_dir
                                                 , cache_dir='../'
                                                 , torch_dtype=torch.float16
                                                 , low_cpu_mem_usage=True
                                                 , device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
        model.lookahead_cache = lookahead_cache
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = None


model_dir = 'your/model/path'
"""
generate answers first if only prompts are available, answers in the warmup samples are used for constructing trie-tree cache
prompt_dir = 'your/prompt/dir'
dataset_dir = 'your/answer/dir'
worker.save_answers(prompt_dir, dataset_dir, prompt_name='your/prompt/field/name', batch_size=1, max_count=None)
"""

# the dataset can be found in lookahead/datasets/dataset.py
dataset_dir = 'dolly_15k_llama2_7b_chat.jsonl'

worker = LlameBenchmark(log_dir='llama_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)
worker.load_prompts(prompt_dir=dataset_dir)

max_length = 256
chat_count = 1000
warmup_count = 10000

# test correctness with lookahead decoding
worker.batch_chat(worker.prompts[:10],
                  max_length=max_length,
                  decoding_length=16,
                  branch_length=4,
                  debug_lookahead=False,
                  erase=True,
                  batch_size=1)


worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  sizes=[64], lens=[0], max_length=max_length)
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  sizes=[64], lens=[12], max_length=max_length)