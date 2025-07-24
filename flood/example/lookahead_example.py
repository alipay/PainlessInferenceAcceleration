# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING']='0'

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

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'

    prompts = []
    for line in open('/ossfs/workspace/tmp/lookahead.jsonl'):
        line = line.strip()
        prompts.append(json.loads(line)['messages'][0]['content'])


    reqs = []
    if False:
        for i, prompt in enumerate(prompts):
            reqs.append(Request(0, input_text=prompt, output_length=512))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        for i, prompt in enumerate(prompts):
            messages = [
                        {"role": "system","content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            reqs.append(Request(i, input_text=text, output_length=512))

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU], record_shapes=False, profile_memory=False, with_flops=True, with_stack=True, with_modules=True) as prof:
    if True:
        worker = LLM(model_path,
                    n_stage=1,  # gpus
                    n_proc=1,
                    chunk_size=1024,
                    # model_dtype=torch.float8_e4m3fn,
                    cache_size=None,
                    slot_fully_alloc_under=1024,
                    tune_alloc_size=False,
                    eos_token_id=None,
                    debug=False,
                    kernels=('sa',),
                    spec_algo = 'lookahead',
                    spec_branch_length=16,
                    max_spec_branch_count=4,
                    logger='example.log')

        # start process
        input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
        worker.launch(input_queue, chunk_queue, working_queue, output_queues)

        # do benchmark
        c = 100
        for i in range(3):
            ts = time.time()
            tokens = 0
            for j in range(c):
                rs = [reqs[j]]
                for k, req in enumerate(worker.request_stream_generate(rs,
                                                            input_queue,
                                                            output_queues,
                                                            print_param=False,
                                                            print_count=0)):
                    # print(f'{i}-{j}')
                    tokens += len(req.output_ids)
                    # print(f'prompt-{i}: ', req.input_text)
                    # print(f'answer-{i}: ', req.output_text)
            te = time.time()
            elapse = te-ts 
            speed = tokens/(te-ts)
            print(f'sample:{c} tokens:{tokens} elapse:{elapse:.3f} speed:{speed:.2f}')
    # print(prof.key_averages().table(sort_by=None, top_level_events_only=True, row_limit=2000))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by=None, row_limit=100))
    # prof.export_chrome_trace("trace.json")

