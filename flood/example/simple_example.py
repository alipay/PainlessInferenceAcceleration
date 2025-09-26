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

    model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    prompts = ['tell me a joke']
    reqs = []
    for i, prompt in enumerate(prompts):
        messages = [
                    {"role": "system","content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # text = prompt
        reqs.append(Request(i, input_text=text, output_length=100))


    worker = LLM(model_path,
                 n_stage=1,  # gpus
                 n_proc=1,
                 chunk_size=1024,
                #  model_dtype=torch.bfloat16,
                 max_concurrency=1024,
                 cache_size=16000,
                 slot_fully_alloc_under=10240,
                 tune_alloc_size=False,
                 eos_token_id=None,
                 debug=False,
                 kernels=('sa',),
                #  spec_algo = 'lookahead',
                 logger='example.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(
        f'\n**********  start benchmark:{time.time() % 1000:.3f}  **********\n')
    for i, req in enumerate(worker.request_stream_generate(reqs,
                                                   input_queue,
                                                   output_queues,
                                                   print_count=0)):
        if i <= 4:
            print('\n\n')
            print(f'prompt-{i}: ', req.input_text.strip('\n'))
            print(f'answer-{i}: ', req.output_text.strip('\n'))
        time.sleep(1.0)
