# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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
    model_path = '/agent/jingyue/moe_lite_linear/v3_convert'

    # do not apply template
    # reqs = [
    #         Request(0, input_text='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nConsider a sequence of real numbers \( a_1, a_2, a_3, \ldots \) defined as follows:\n \( a_1 = 1 \)\n For \( n \geq 1 \), \( a_{n+1} = \frac{a_n + 2}{a_n + 1} \).\nDetermine the value of \( a_{2024} \).<|im_end|>\n<|im_start|>assistant\n', output_length=2048),
    #         Request(1, input_text='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nConsider the polynomial \( P(x, y) = x^4 + y^4 - 2x^2y^2 \). Let \( S \) be the set of points \( (x, y) \) where \( x \) and \( y \) are integers in the range \( -10 \leq x, y \leq 10 \) such that \( P(x, y) = 0 \). Determine the number of elements in the set \( S \).<|im_end|>\n<|im_start|>assistant\n', output_length=2048)
    #         ]
    # apply template

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # prompts = ['Consider a sequence of real numbers \( a_1, a_2, a_3, \ldots \) defined as follows: \( a_1 = 1 \)\n For \( n \geq 1 \), \( a_{n+1} = \\frac{a_n + 2}{a_n + 1} \).\nDetermine the value of \( a_{2024} \)']
    prompts = ['杭州在哪里']
    reqs = []
    for i, prompt in enumerate(prompts):
        messages = [
                    {"role": "system","content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        reqs.append(Request(i, input_text=text, output_length=128))


    worker = LLM(model_path,
                 n_stage=1,  # gpus
                 n_proc=1,
                 chunk_size=4096,
                #  model_dtype=torch.float8_e4m3fn,
                 max_concurrency=1024,
                 cache_size=16000,
                 slot_fully_alloc_under=1024,
                 tune_alloc_size=False,
                 eos_token_id=None,
                 debug=True,
                 kernels=('fa2',),
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
