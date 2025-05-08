# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

import random
import time
import torch
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoTokenizer
from flood.facade.llm import LLM
from flood.utils.request import Request
from flood.utils.reader import Reader

random.seed(7)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'
    # model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'
    # model_path = '/mntnlp/nanxiao/model'
    # model_path = '/mntnlp/nanxiao/deepseekv3'
    # model_path = '/agent/nanxiao/models/Qwen2.5-32B-Instruct'
    model_path = '/mnt/nas_acr89/jingyue/deepseekv3'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    prompts = ['杭州在哪里']
    reqs = []
    for i, prompt in enumerate(prompts):
        messages = [
                    # {"role": "system","content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        reqs.append(Request(i, input_text=text, output_length=128))


    worker = LLM(model_path,
                 n_stage=2,  # gpus
                 n_proc=3,
                 chunk_size=1024,
                #  model_dtype=torch.float8_e4m3fn,
                 max_concurrency=1024,
                 cache_size=16000,
                 slot_fully_alloc_under=1024,
                 tune_alloc_size=False,
                 eos_token_id=None,
                 debug=True,
                 kernels=('mla',),
                #  spec_algo = 'lookahead',
                 logger='example.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    for i, req in enumerate(worker.request_stream_generate(reqs,
                                                input_queue,
                                                output_queues,
                                                print_count=0)):
        print('\n\n')
        print(f'prompt-{i}: ', req.input_text)
        print(f'answer-{i}: ', req.output_text)
    time.sleep(1.0)
