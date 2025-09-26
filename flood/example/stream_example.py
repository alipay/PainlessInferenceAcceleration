# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch.multiprocessing as mp

from flood.facade.llm import LLM
from flood.utils.request import Request

random.seed(7)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    model_path = "/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct"

    worker = LLM(
        model_path,
        n_stage=1,
        n_proc=1,
        schedule_mode="timely",
        eos_token_id=None,
        kernels=("sa",),
        spec_algo="lookahead",
        spec_branch_length=8,
        max_spec_branch_count=8,
        debug=False,
        logger="stream.log",
    )

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    request = Request("rid", input_text="hi!", output_length=200)
    for seg in worker.stream_generate(request, input_queue, output_queues):
        print(seg, end="", flush=True)
