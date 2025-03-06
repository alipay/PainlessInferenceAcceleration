# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = '1'
# os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:24,roundup_power2_divisions:[2:1,4:2,8:4,16:8,>:16],garbage_collection_threshold:0.99'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'roundup_power2_divisions:[2:1,4:2,8:4,16:8,>:16]'

import random
import math
import time
import json
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp

from flood.utils.reader import Reader
from flood.common.llm import LLM
from flood.utils.request import Request

random.seed(7)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    model_path = '/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct'  # llama
    # model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct' # llama
    # model_path = '/mnt/prev_nas/nanxiao/llama3'  # llama(empty)
    # model_path = '/mnt/nas_acr89/jingyue/med-fp8'  # llama(fp8 dynamic)
    # model_path = '/mnt/nas_acr89/jingyue/med-fp8static'  # llama(fp8 static)
    # model_path = '/mnt/nas_acr89/nanxiao/chat_80b'  # bailing80b
    # model_path = "/mnt/nas1b/user/zhaoxin/llm_pretrain/model_hub/bailing-1b/bailing_1B_Base_newformart"  # bailing1b
    # model_path = '/mnt/prev_nas/nanxiao/bailing'  # bailing(empty)
    # model_path = '/mnt/nas_acr89/jingyue/med-int8-1024'
    # model_path = '/mnt/nas_acr89/jingyue/med-int8static'
    # model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-0.5B-Instruct'  # qwen
    # model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-14B-Instruct'  # qwen
    # model_path = '/naseve/user/zhaoxin/llm_pretrain/model_hub/Qwen2.5-7B'
    # model_path = '/mnt/nas_sgk32/jingyue/Bailing-4.0-MoE-Plus_A29B-4K-Chat-20241120-DeepSeek'
    # model_path = '/mnt/nas_sgk32/jingyue/Bailing-4.0-MoE-Plus_A29B-4K-Chat-20241120-DeepSeek-dummy'
    # model_path = '/mnt/nas_sgk32/jingyue/Bailing-4.0-MoE-Plus_A29B-4K-Chat-20241120-Deepseek-fp8'
    # model_path = '/mnt/nas_acr89/jingyue/moe-lite-fp8'
    # model_path = '/mnt/zhuli_nas/lyn.zyl/models/bailing_format_models/Bailing-4.0-10B-16K-Chat-20241029'
    # model_path = '/mnt/nas_acr89/jingyue/bailing-moe-lite' # bailingmoe(deepseek format)
    # model_path = '/mnt/nas_acr89/jingyue/moe-lite'  # bailingmoe(deepseek format)
    # model_path = '/mnt/nas_acr89/jingyue/bailing-moe-lite'  # bailingmoe
    # model_path = '/agent/jingyue/moe_lite_base-bailing-safetensor'

    # reqs = Reader.read_fix_dataset(model_path, max_count=4, output_length=100)

    reqs = Reader.read_dummy_dataset(max_count=20000, input_length=100, output_length=300, flunc=0.1)

    # data_path = '/mntnlp/nanxiao/dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'
    # reqs = Reader.read_sharegpt_dataset(data_path, model_path, max_count=20000)


    # load model
    n_stage = 1
    n_proc = 2
    eos_token_id = ()
    worker = LLM(model_path,
                #  model_dtype=torch.float8_e4m3fn,
                #  head_dtype=torch.float8_e4m3fn,
                #  emb_dtype=torch.float8_e4m3fn,
                #  cache_dtype=torch.float8_e4m3fn,
                 n_stage=n_stage,
                 n_proc=n_proc,
                 cache_size=None,
                 slot_size=8192,
                 schedule_mode='pingpong',
                 chunk_size=1024,
                 sync_wait_time=(4.0,4.0),
                 queue_timeout=0.0005,
                 max_slot_alloc_fail_count=2,
                 alloc_early_exit_rate=0.95,
                 slot_first_alloc_rate=1.0,
                 slot_fully_alloc_under=256,
                 min_batch_size=16,
                 max_batch_size=512,
                 batch_size_step=None,
                 batch_size_round_frac=0.0,  # 0.585
                 min_decode_rate=1.0, # 0.8
                 eos_token_id=eos_token_id,
                 kernels=('sa',),
                 output_file_name='tmp.jsonl',
                 output_file_mode='w+',
                 logger='ignore.log',
                 debug=True)

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(f'\n****************  start benchmark:{time.time()%1000:.3f}  *******************\n')
    for i, req in  enumerate(worker.request_stream_generate(reqs, 
                    input_queue, 
                    output_queues,
                    print_count=0)):
        if i <= 1:
            # print(f'\n\nprompt-{i}: ', req.input_text.replace('\n','\\n'))
            print(f'answer-{i}: ', req.output_text.replace('\n','\\n'),'\n\n')



