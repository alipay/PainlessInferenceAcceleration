# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch.multiprocessing as mp

from flood.common.llm import LLM
from flood.utils.request import Request as FloodRequest

random.seed(7)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # model_path = '/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct'  # llama
    # model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct' # llama
    # model_path = '/mnt/nas_acr89/nanxiao/chat_80b'  # bailing
    model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-0.5B-Instruct'

    pred_path = 'tmp.jsonl'

    # load 8b model with 2*A100
    worker = LLM(model_path,
                 n_stage=1,
                 n_proc=1,
                 schedule_mode='low_latency',
                 eos_token_id=None,
                 debug=False,
                 output_file_name=pred_path,
                 output_file_mode='w+',
                 output_field_names=('output_text',),
                 logger='antoc.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    request = FloodRequest('rid', input_text='hi!', output_length=1000)
    for seg in worker.stream_generate(request, input_queue, output_queues):
        print(seg, end='', flush=True)
