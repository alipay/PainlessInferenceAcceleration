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

    model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'

    worker = LLM(model_path,
                 n_stage=1,
                 n_proc=1,
                 schedule_mode='timely',
                 eos_token_id=None,
                 debug=False,
                 output_file_name='tmp.jsonl',
                 output_file_mode='w+',
                 output_field_names=('output_text',),
                 logger='stream.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    request = FloodRequest('rid', input_text='hi!', output_length=1000)
    for seg in worker.stream_generate(request, input_queue, output_queues):
        print(seg, end='', flush=True)
