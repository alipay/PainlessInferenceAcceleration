# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import random
import time

import torch
import torch.multiprocessing as mp

from flood.common.llm import LLM
from flood.utils.reader import Reader

random.seed(7)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'
    # reqs = Reader.read_fix_dataset(model_path, max_count=1, output_length=100)

    reqs = Reader.read_dummy_dataset(max_count=10000, input_length=128,
                                     output_length=128, flunc=0.1)

    # data_path = 'your/path/ShareGPT_V3_unfiltered_cleaned_split.json'
    # reqs = Reader.read_sharegpt_dataset(data_path, model_path, max_count=10000)

    # load model
    n_stage = 1
    n_proc = 2
    eos_token_id = ()  # set eos_token_id = () to make it ignore eos
    worker = LLM(model_path,
                 model_dtype=torch.bfloat16,
                 head_dtype=torch.bfloat16,
                 emb_dtype=torch.bfloat16,
                 cache_dtype=torch.bfloat16,
                 n_stage=n_stage,
                 n_proc=n_proc,
                 cache_size=None,
                 slot_size=8192,
                 schedule_mode='pingpong',
                 chunk_size=1024,
                 sync_wait_time=(4.0, 4.0),
                 queue_timeout=0.0005,
                 max_slot_alloc_fail_count=1,
                 alloc_early_exit_rate=0.95,
                 slot_first_alloc_rate=1.0,
                 slot_fully_alloc_under=1000,
                 min_batch_size=16,
                 max_batch_size=512,
                 batch_size_step=None,
                 batch_size_round_frac=0.0,  # 0.585
                 min_decode_rate=1.0,  # 0.8
                 eos_token_id=eos_token_id,
                 output_file_name='tmp.jsonl',
                 output_file_mode='w+',
                 logger='benchmark.log',
                 debug=True)

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(
        f'\n*********  start benchmark:{time.time() % 1000:.3f}  ***********\n')
    for i, req in enumerate(worker.request_stream_generate(reqs,
                                                   input_queue,
                                                   output_queues,
                                                   print_count=0)):
        if i <= 3:
            print(f'\n\nprompt-{i}: ', req.input_text.replace('\n', '\\n'))
            print(f'answer-{i}: ', req.output_text.replace('\n', '\\n'), '\n\n')
