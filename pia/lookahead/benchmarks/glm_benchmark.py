# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import sys
import torch
from transformers import AutoTokenizer

from pia.lookahead.common.lookahead_cache import LookaheadCache
from benchmark import Benchmark


class GlmBenchmark(Benchmark):

    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from models.glm.modeling_glm import GLMForConditionalGeneration
        from models.glm.tokenization_glm import GLMChineseTokenizer
        model = GLMForConditionalGeneration.from_pretrained(model_dir
                                                            , cache_dir='../'
                                                            , offload_folder='./'
                                                            , torch_dtype=torch.float16
                                                            , low_cpu_mem_usage=True
                                                            , device_map='auto')
        tokenizer = GLMChineseTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token

        # stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        # lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
        lookahead_cache = LookaheadCache(eos=50005, stop_words={43359, 43360, 43361, 43362})
        model.lookahead_cache = lookahead_cache
        self.model = model
        self.tokenizer = tokenizer
        self.eos = 50005
        self.eop = 50006

    def tokenize(self, prompt, max_length=256):
        tokenizer = self.tokenizer
        if isinstance(prompt, list):
            prompt = [x if '[gMASK]' in x else x + '[gMASK]' for x in prompt]
            inputs = tokenizer(prompt,
                               padding=True,
                               truncation=False,
                               return_tensors="pt")
        else:
            if '[gMASK]' not in prompt:
                prompt = prompt + '[gMASK]'
            inputs = tokenizer(prompt,
                               padding=True,
                               truncation=False,
                               return_tensors="pt",
                               )

        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=max_length)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['generation_attention_mask'].to(self.device)
        position_ids = inputs['position_ids'].to(self.device)
        return input_ids, position_ids, attention_mask


model_dir = 'your/model/path'
"""
generate answers first if only prompts are available, answers in the warmup samples are used for constructing trie-tree cache
prompt_dir = 'your/prompt/dir'
dataset_dir = 'your/answer/dir'
worker.save_answers(prompt_dir, answer_dir, prompt_name='your/prompt/field/name', batch_size=1, max_count=None)
"""
dataset_dir = 'your/dataset/path'

worker = GlmBenchmark(log_dir='antglm_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)
worker.load_prompts(prompt_dir=dataset_dir)

max_length = 256
chat_count = 1000
warmup_count = 10000

# test correctness with lookahead decoding
worker.batch_chat(worker.prompts[:10],
                  max_length=max_length,
                  decoding_length=15,
                  branch_length=4,
                  debug_lookahead=False,
                  erase=True,
                  batch_size=1)

# performance check
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  sizes=[64], lens=[0], max_length=max_length)
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count],
                  sizes=[64], lens=[12], max_length=max_length)