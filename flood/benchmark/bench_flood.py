# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os 
import random
import time

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from flood.facade.llm import LLM
from flood.utils.reader import Reader
from flood.utils.request import Request

random.seed(7)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct'
    # model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'
    # model_path = '/mnt/prev_nas/chatgpt/pretrained_models/Qwen2.5-72B-Instruct'
    model_path = '/agent/nanxiao/models/Qwen2.5-32B-Instruct'
    # reqs = Reader.read_fix_dataset(model_path, max_count=1, output_length=100)

    # reqs = Reader.read_dummy_dataset(max_count=20000, input_length=100,
    #                                  output_length=300, flunc=0.1)

    # data_path = '/mntnlp/nanxiao/dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'
    # reqs = Reader.read_sharegpt_dataset(data_path, model_path, max_count=20000)


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    reqs = []
    for i, line in enumerate(open('/ossfs/workspace/tmp/kag.txt')):
        chat = [{"role": "user", "content": line.strip()}]
        prompt = tokenizer.apply_chat_template(chat, 
                                               tokenize=False,
                                               add_generation_prompt=True)
        request = Request(i, input_text=prompt, input_length=0,
                        output_length=4096)
        reqs.append(request)


    # load model
    n_stage = 4
    n_proc = 5
    eos_token_id = None  # set eos_token_id = () to make it ignore eos
    worker = LLM(model_path,
                #  model_dtype=torch.bfloat16,
                #  head_dtype=torch.bfloat16,
                #  emb_dtype=torch.bfloat16,
                #  cache_dtype=torch.bfloat16,
                 n_stage=n_stage,
                 n_proc=n_proc,
                #  cache_size=0.80,
                 slot_count=8192,
                 schedule_mode='pingpong',
                 chunk_size=1024,
                 sync_wait_time=(4.0, 4.0),
                 queue_timeout=0.0005,
                 max_slot_alloc_fail_count=4,
                 alloc_early_exit_rate=0.95,
                 slot_fully_alloc_under=1024,
                 tune_alloc_size=True,
                 min_batch_size=16,
                 max_batch_size=512,
                 batch_size_step=128,  # H20:128 L20:128 A100:64
                 batch_size_round_frac=0.0,  # 0.585
                 min_decode_rate=1.0,  # 0.8
                 kernels=('sa',),
                 eos_token_id=eos_token_id,
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
                                                   print_count=3)):
        if i <= -1:
            print(f'\n\nprompt-{i}: ', req.input_text.replace('\n', '\\n'))
            print(f'answer-{i}: ', req.output_text.replace('\n', '\\n'), '\n\n')
