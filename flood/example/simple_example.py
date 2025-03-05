# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import random
import time

import torch.multiprocessing as mp

from flood.common.llm import LLM
from flood.utils.request import Request

random.seed(7)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # model_path = '/mntnlp/nanxiao/model'
    model_path = '/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct'  # llama8b
    # model_path = '/mnt/nas_acr89/nanxiao/chat_80b'  # bailing80b
    # model_path = '/agent/jingyue/Bailing-4.0-MoE-Lite_A2.8B-16K-Base-20240914-bailing'

    pred_path = 'tmp.jsonl'  # 推理结果文件

    """
    构造批次推理请求
    rid/input_text/output_length是必须要的
    rid: 请求id， 必须
    input_text: prompt, 必须
    output_length: 最大生成长度，够用的前提下越小越好
    """
    # reqs = [Request(0, input_text='tell me a joke.', output_length=32)]  
    reqs = [Request(0, input_text='tell me a joke.', output_length=64),
            Request(1, input_text='make me laugh.', output_length=64)]
    # reqs = [Request(0, input_text='tell me a joke.', output_length=100, temperature=0.9, top_p=0.95, top_k=20, min_p=0.3), Request(1, input_text='make me laugh.', output_length=100)]  

    worker = LLM(model_path,
                 n_stage=1,  # gpus
                 n_proc=1,
                 eos_token_id=None,
                 debug=True,
                 alloc_early_exit_rate=0.95,
                 slot_first_alloc_rate=0.5,
                 slot_fully_alloc_under=8,
                 output_file_name=pred_path,
                 output_file_mode='w+',
                 logger='example.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(
        f'\n****************  start benchmark:{time.time() % 1000:.3f}  *******************\n')
    responses = worker.generate(reqs,
                                input_queue,
                                output_queues,
                                print_count=10)
