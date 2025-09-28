# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import argparse
import os
import random
import time

import torch
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from flood.facade.dist_llm import DistLLM
from flood.utils.request import Request

random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--rank", "--rank", type=int, default=0)
parser.add_argument("--world-size", "--world-size", type=int, default=2)
parser.add_argument("--master", "--master", type=str, default="127.0.0.1")
parser.add_argument("--port", "--port", type=str, default="40000")
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANK = args.rank
WORLD_SIZE = args.world_size
MASTER = args.master
PORT = args.port
os.environ["FLOOD_RANK"] = str(RANK)
os.environ["FLOOD_WORLD_SIZE"] = str(WORLD_SIZE)
os.environ["FLOOD_MASTER"] = MASTER
os.environ["FLOOD_PORT"] = PORT

# print(f'{RANK=} {WORLD_SIZE=} {MASTER_ADDR=} {MASTER_PORT=}')

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_CHECKS_DISABLE"] = "1"
# os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
# os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "0"

# USAGE
# on node 1:  python /076074/framework/example/dist_example.py
# --master=ip --port=40000 --world-size=2 --rank=0
# on node 2:  python /076074/framework/example/dist_example.py
# --master=ip --port=40000 --world-size=2 --rank=1

# NOTE: this is an experimental feature, may contain bugs.


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    model_path = "/mntnlp/common_base_model/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    WORLD_SIZE = int(os.environ["FLOOD_WORLD_SIZE"])
    RANK = int(os.environ["FLOOD_RANK"])
    MASTER = os.environ["FLOOD_MASTER"]
    PORT = int(os.environ["FLOOD_PORT"])
    print(f"{MASTER=} {PORT=} {WORLD_SIZE=} {RANK=}")

    pred_path = "tmp.jsonl"

    prompts = ["1 + 1 = ?", "tell me a joke"]

    reqs = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": " You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        reqs.append(Request(i, input_text=text, output_length=4096))

    print("start init LLM")
    worker = DistLLM(
        model_path,
        #  cache_dtype=torch.bfloat16,
        n_stage=2,  # gpu count
        n_proc=3,  # process count
        cache_size=0.9,
        #  eos_token_id=(),
        debug=True,
        max_concurrency=1024,
        kernels=("sa",),
        slot_fully_alloc_under=4096,
        tune_alloc_size=False,
        logger="dist.log",
    )
    print("finish init LLM")

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    if RANK == 0:
        # do benchmark
        print(f"\n*********  start benchmark:{time.time() % 1000:.3f}  *********\n")
        responses = worker.generate(reqs, input_queue, output_queues, print_count=10)
    else:
        while True:
            time.sleep(0.001)
