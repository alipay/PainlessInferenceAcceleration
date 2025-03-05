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
        from pia.lookahead.models.glm.modeling_glm_batch import GLMForConditionalGeneration
        from pia.lookahead.models.glm.tokenization_glm import GLMChineseTokenizer
        model = GLMForConditionalGeneration.from_pretrained(model_dir
                                                            , torch_dtype=torch.float16
                                                            , low_cpu_mem_usage=True
                                                            , device_map={"":"cuda:0"})
        tokenizer = GLMChineseTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        model.lookahead_cache = LookaheadCache()
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eop_token_id
        self.eop = 50006
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words+['的','是'])

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


model_dir = '/mntnlp/nanxiao/lookahead/antglm'
worker = GlmBenchmark(log_dir='antglm_benchmark')
worker.initialize(model_dir=model_dir, token_dir=model_dir)

"""
answers in the warmup samples are used for constructing trie-tree cache
"""
# warmup_prompt_dir = '/mntnlp/nanxiao/dataset/antrag_8k/train.jsonl'
# warmup_dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/train.jsonl'
# worker.save_answers(warmup_prompt_dir, warmup_dataset_dir, batch_size=1, max_count=None, use_lookahead=True)

# dataset_dir = 'your/dataset/path'
dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/test.jsonl'
warmup_dataset_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/train.jsonl'
worker.load_prompts(prompt_dir=dataset_dir, warmup_prompt_dir=warmup_dataset_dir)

# test correctness with lookahead decoding
worker.batch_chat(worker.prompts[:1],
                  max_new_tokens=256,
                  decoding_length=16,
                  branch_length=4,
                  debug_lookahead=False,
                  erase=True,
                  batch_size=1)

# performance check
max_new_tokens = 256
chat_count = 1000
warmup_count = 10000
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.warmup_ids[:warmup_count],
                  sizes=[128], lens=[32], max_new_tokens=max_new_tokens)