# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random
import time

import torch.multiprocessing as mp

from flood.common.llm import LLM
from flood.utils.reader import Reader

random.seed(7)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'

    # read prompt
    reqs = Reader.read_dummy_dataset(model_path, max_count=1000,
                                     input_length=500, output_length=200,
                                     flunc=0.1)
    if len(reqs) == 0:
        exit()
    reqs = Reader.sort_by(reqs, key='random')
    for req in reqs:
        req.rid = str(req.rid)
        req.emb_idx = 1
        req.emb_size = 256

    # load model
    worker = LLM(model_path,
                 n_stage=1,
                 n_proc=1,
                 cache_size=0.9,
                 slot_size=8192,
                 schedule_mode='pingpong',
                 chunk_size=1024,
                 sync_wait_time=(4.0, 4.0),
                 queue_timeout=0.001,
                 max_slot_alloc_fail_count=1,
                 alloc_early_exit_rate=0.98,
                 slot_rate=1.0,
                 min_batch_size=16,
                 max_batch_size=512,
                 batch_size_step=None,
                 batch_size_round_frac=0.0,  # 0.585
                 min_decode_rate=1.0,
                 eos_token_id=None,
                 debug=True,
                 output_file_name='tmp.jsonl',
                 output_file_mode='w+',
                 embedding_dir='embs.safetensors',
                 logger='multimodel.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(
        f'\n****************  start benchmark:{time.time() % 1000:.3f}  *******************\n')
    worker.generate(reqs, input_queue, output_queues, print_count=10)
