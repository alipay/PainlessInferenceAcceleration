# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import random
import time
import copy
import json
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from flood.facade.llm import LLM
from flood.utils.reader import Reader
from flood.utils.request import Request

random.seed(7)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # model_path = "/mntnlp/common_base_model/Llama-3.1-8B-Instruct"
    model_path = "inclusionAI/Ring-mini-linear-2.0"

    warmup_reqs = Reader.read_dummy_dataset(
        max_count=64, input_length=64, output_length=8, flunc=0.1
    )

    # data_path = 'your/path/ShareGPT_V3_unfiltered_cleaned_split.json'
    # reqs = Reader.read_sharegpt_dataset(data_path, model_path, max_count=10000)

    data_path = "/mntnlp/nanxiao/dataset/gsm_8k/test.jsonl"
    reqs = Reader.read_jsonl_dataset(
        data_path, model_path, prompt_name="prompt", max_count=1024, output_length=4096
    )

    # load model
    n_stage = 1
    n_proc = 1
    eos_token_id = None  # ()  # set eos_token_id = () to make it ignore eos
    worker = LLM(
        model_path,
        #  model_dtype=torch.bfloat16,
        #  head_dtype=torch.bfloat16,
        #  emb_dtype=torch.bfloat16,
        #  cache_dtype=torch.bfloat16,
        n_stage=n_stage,
        n_proc=n_proc,
        #  cache_size=0.9,
        slot_count=8192,
        max_concurrency=128,
        schedule_mode="pingpong",
        chunk_size=1024,
        sync_wait_time=(4.0, 4.0),
        queue_timeout=0.001,
        max_slot_alloc_fail_count=1,
        alloc_early_exit_rate=0.95,
        slot_fully_alloc_under=10240,
        max_extend_size=256,
        tune_alloc_size=False,
        min_batch_size=16,
        max_batch_size=1024,
        batch_size_step=64,  # H20:128 L20:128 A100:64
        batch_size_round_frac=0.0,  # 0.585
        min_decode_rate=0.8,  # 0.8
        kernels=("sa",),
        # spec_algo='lookahead',
        # max_spec_branch_count=4,
        # spec_branch_length=4,
        # spec_token_budget_count=128,
        # use_spec_min_batch_size=32,
        eos_token_id=eos_token_id,
        logger="benchmark.log",
        debug=True,
    )

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # warmup
    worker.generate(warmup_reqs, input_queue, output_queues, print_count=0)
    time.sleep(1)

    # do benchmark
    print(f"\n*********  start benchmark:{time.time() % 1000:.3f}  ***********\n")
    for i, req in enumerate(
        worker.request_stream_generate(
            reqs[:128], input_queue, output_queues, print_count=0
        )
    ):
        if i <= -1:
            print("\n\n")
            print(f"prompt-{i}: ", req.input_text)
            print(f"answer-{i}: ", req.output_text)
