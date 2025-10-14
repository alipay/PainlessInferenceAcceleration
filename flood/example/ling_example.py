# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import json
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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    model_path = "inclusionAI/Ring-mini-linear-2.0"

    prompts = ["Tell me a joke"] + [
        "Give me a short introduction to large language models."
    ] * 32

    reqs = []

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        reqs.append(Request(i, input_text=text, output_length=512))

    worker = LLM(
        model_path,
        n_stage=1,  # gpus
        n_proc=1,
        chunk_size=1024,
        # model_dtype=torch.float8_e4m3fn,
        cache_size=None,
        max_concurrency=64,
        slot_fully_alloc_under=1024,
        tune_alloc_size=False,
        eos_token_id=None,
        debug=False,
        kernels=("sa",),
        spec_algo="lookahead",
        spec_branch_length=4,
        max_spec_branch_count=2,
        spec_token_budget_count=256,
        use_spec_min_batch_size=32,
        logger="example.log",
    )

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(f"\n**********  start benchmark:{time.time() % 1000:.3f}  **********\n")
    worker.request_stream_generate(reqs[:1], input_queue, output_queues, print_count=0)

    for _ in range(5):
        ts = time.time()
        tokens = 0
        for i, req in enumerate(
            worker.request_stream_generate(
                reqs[1:], input_queue, output_queues, print_count=0
            )
        ):
            tokens += len(req.output_ids)
            if i <= -1:
                print("\n\n")
                print(f"prompt-{i}: ", req.input_text.strip("\n"))
                print(f"answer-{i}: ", req.output_text.strip("\n"))
        te = time.time()
        elapse = te - ts
        speed = tokens / (te - ts)
        print(
            f"sample:{len(reqs)} tokens:{tokens} elapse:{elapse:.3f} speed:{speed:.2f}"
        )
