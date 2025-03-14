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

    model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'

    reqs = [Request(0, input_text='tell me a joke.', output_length=64),
            Request(1, input_text='make me laugh.', output_length=64)]

    worker = LLM(model_path,
                 n_stage=1,  # gpus
                 n_proc=1,
                 eos_token_id=None,
                 debug=True,
                 output_file_name='tmp.jsonl',
                 output_file_mode='w+',
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
