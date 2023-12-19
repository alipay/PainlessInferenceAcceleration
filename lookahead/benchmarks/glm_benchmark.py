# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import sys
sys.path.append('..')
import torch
from common.lookahead_cache import LookaheadCache

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import AutoTokenizer

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


model_dir = '/mntnlp/nanxiao/lookahead_benchmark/antrag'
prompt_dir = '/mntnlp/nanxiao/lookahead_benchmark/search_8k_antglm_10b.jsonl'

worker = GlmBenchmark(log_dir='antglm_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)
worker.load_prompts(prompt_dir=prompt_dir)

prompt = '杭州在哪里？'
max_length = 256
chat_count = 1000
warmup_count = 10000

# runable check
worker.chat(prompt,
            max_length=max_length,
            use_lookahead=False,
            decoding_length=15,
            branch_length=4,
            debug_lookahead=False)

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