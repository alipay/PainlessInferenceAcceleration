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

from transformers import AutoTokenizer
from flood.facade.llm import LLM
from flood.utils.request import Request
from flood.utils.reader import Reader

random.seed(7)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'
    # model_path = '/mnt/prev_nas/chatgpt/pretrained_models/Qwen2.5-7B-Instruct'
    # model_path = '/mntnlp/nanxiao/model'
    # model_path = '/mntnlp/nanxiao/deepseekv3'
    model_path = '/agent/nanxiao/models/Qwen2.5-32B-Instruct'

    # without template
    reqs = [
            Request(0, input_text='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n**Lily\'s Flower Garden** Lily has a small flower garden where she plants tulips and daisies. One day, she decides to count the flowers. She finds that she has 24 tulips and 18 daisies. She wants to plant them in her garden in rows, with each row containing the same number of flowers and with no flowers left over. What is the greatest number of flowers she can plant in each row without mixing tulips and daisies in the same row?<|im_end|>\n<|im_start|>assistant\n', output_length=4096),
            Request(1, input_text='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nConsider the polynomial \( P(x, y) = x^4 + y^4 - 2x^2y^2 \). Let \( S \) be the set of points \( (x, y) \) where \( x \) and \( y \) are integers in the range \( -10 \leq x, y \leq 10 \) such that \( P(x, y) = 0 \). Determine the number of elements in the set \( S \).<|im_end|>\n<|im_start|>assistant\n', output_length=4096)
            ]
    # with template
    # reqs = Reader.read_fix_dataset(model_path, prompts=['Consider a sequence of positive integers \( a_1, a_2, a_3, \ldots, a_n \) such that each term is the sum of the distinct positive integers less than it. For example, \( a_5 \) would be \( a_1 + a_2 + a_3 + a_4 \) since 1, 2, 3, and 4 are all the distinct positive integers less than 5. \nDefine \( S(n) \) as the sum of the sequence \( a_1, a_2, a_3, \ldots, a_n \). For instance, \( S(5) = a_1 + a_2 + a_3 + a_4 + a_5 \).\nGiven that \( a_1 = 1 \) and \( a_n = a_{n-1} + a_{n-2} + \cdots + a_1 \) for \( n \geq 2 \), find the value of \( S(6) \).'], output_length=512)

    worker = LLM(model_path,
                 n_stage=1,  # gpus
                 n_proc=1,
                 chunk_size=128,
                #  model_dtype=torch.float8_e4m3fn,
                 cache_size=16000,
                 slot_fully_alloc_under=1024,
                 tune_alloc_size=True,
                 eos_token_id=None,
                 debug=True,
                 kernels=('sa',),
                #  spec_algo = 'lookahead',
                 logger='example.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(
        f'\n**********  start benchmark:{time.time() % 1000:.3f}  **********\n')
    responses = worker.generate(reqs,
                                    input_queue,
                                    output_queues,
                                    print_count=10)
