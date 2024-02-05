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
        # org version llama
        # from pia.lookahead.models.llama.modeling_llama import LlamaForCausalLM
        # fused op version llama
        from pia.lookahead.models.llama.modeling_llama_batch import LlamaForCausalLM 
        tokenizer = AutoTokenizer.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = LlamaForCausalLM.from_pretrained(model_dir
                                                 , cache_dir='../'
                                                 , torch_dtype=torch.float16
                                                 , low_cpu_mem_usage=True
                                                 , device_map='auto')
        model.lookahead_cache = LookaheadCache()
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = None


model_dir = '/mntnlp/common_base_model/llama2-13b-chat'
worker = LlameBenchmark(log_dir='llama_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)

# answers in the warmup samples are used for constructing trie-tree cache
warmup_prompt_dir = '/mntnlp/nanxiao/dataset/dolly_15k/train.jsonl'
warmup_dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/dolly_15k_llama2_7b_chat/train_v100.jsonl'
worker.save_answers(warmup_prompt_dir, warmup_dataset_dir, batch_size=1, max_count=1000, use_lookahead=False)

# the dataset can be found in lookahead/datasets/dataset.py
dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/dolly_15k_llama2_13b_chat/test.jsonl'
warmup_dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/dolly_15k_llama2_13b_chat/train_v100.jsonl'
worker.load_prompts(prompt_dir=dataset_dir, warmup_prompt_dir=warmup_dataset_dir)

# test correctness with lookahead decoding
worker.batch_chat(worker.prompts[:3],
                  max_new_tokens=256,
                  decoding_length=16,
                  branch_length=4,
                  debug_lookahead=False,
                  erase=True,
                  batch_size=1)

max_new_tokens = 256
chat_count = 100
warmup_count = 10000
# worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.warmup_ids[:warmup_count],
#                   sizes=[64], lens=[0], max_new_tokens=max_new_tokens)

worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.warmup_ids[:warmup_count],
                  sizes=[16,24,32], lens=[6,8], max_new_tokens=max_new_tokens, decoding_mode='hier',max_query_length=3)