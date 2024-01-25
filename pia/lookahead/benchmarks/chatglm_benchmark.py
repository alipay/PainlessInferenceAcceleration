# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import sys
import torch
from transformers import AutoTokenizer

from pia.lookahead.common.lookahead_cache import LookaheadCache
from benchmark import Benchmark


class ChatglmBenchmark(Benchmark):

    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from pia.lookahead.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
        from pia.lookahead.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
        model = ChatGLMForConditionalGeneration.from_pretrained(model_dir
                                                 , cache_dir='../'
                                                 , torch_dtype=torch.float16
                                                 , low_cpu_mem_usage=True
                                                 , device_map='auto')
        tokenizer = ChatGLMTokenizer.from_pretrained(token_dir)
        stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        model.lookahead_cache = LookaheadCache(stop_words=stop_ids)
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = 50006

    def tokenize(self, prompt, max_length=256):
        inputs = self.model.build_inputs(self.tokenizer, prompt, history=[])
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = None
        return input_ids, position_ids, attention_mask


model_dir = 'your/model/path'
"""
generate answers first if only prompts are available, answers in the warmup samples are used for constructing trie-tree cache
prompt_dir = 'your/prompt/dir'
dataset_dir = 'your/dataset/dir'
worker.save_answers(prompt_dir, dataset_dir, prompt_name='your/prompt/field/name', batch_size=1, max_count=None)
"""
dateset_dir = 'your/dataset/path'

worker = ChatglmBenchmark(log_dir='chatglm_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)
worker.load_prompts(prompt_dir=dateset_dir)

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