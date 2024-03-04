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
        model.lookahead_cache = LookaheadCache()
        tokenizer = ChatGLMTokenizer.from_pretrained(token_dir)
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
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



model_dir = '/mntnlp/common_base_model/chatglm2'
worker = ChatglmBenchmark(log_dir='chatglm_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)

# warmup_prompt_dir = '/mntnlp/nanxiao/dataset/dolly_15k/train.jsonl'
# warmup_dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/dolly_15k_chatglm2/train.jsonl'
# worker.save_answers(warmup_prompt_dir, warmup_dataset_dir, batch_size=1, max_count=None, use_lookahead=True)

# the dataset can be found in lookahead/datasets/dataset.py
dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/dolly_15k_chatglm2/test.jsonl'
warmup_dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/dolly_15k_chatglm2/train.jsonl'
worker.load_prompts(prompt_dir=dataset_dir, warmup_prompt_dir=warmup_dataset_dir)


# test correctness with lookahead decoding
worker.batch_chat(worker.prompts[:1],
                  max_new_tokens=256,
                  decoding_length=16,
                  branch_length=4,
                  debug_lookahead=False,
                  erase=True,
                  batch_size=1)

max_new_tokens = 256
chat_count = 1000
warmup_count = 10000
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.warmup_ids[chat_count:chat_count + warmup_count],
                  sizes=[64], lens=[12], max_new_tokens=max_new_tokens)