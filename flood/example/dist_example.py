# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import argparse
import os
import random
import time

import torch.multiprocessing as mp

from flood.common.dist_llm import DistLLM
from flood.utils.request import Request

random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--rank", "--rank", type=int, default=0)
parser.add_argument("--world-size", "--world-size", type=int, default=2)
parser.add_argument("--master", "--master", type=str, default='127.0.0.1')
parser.add_argument("--port", "--port", type=str, default='40000')
args = parser.parse_args()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# RANK = int(os.environ['RANK'])
# WORLD_SIZE = int(os.environ['WORLD_SIZE'])
# MASTER_ADDR = os.environ['MASTER_ADDR']
# MASTER_PORT = os.environ['MASTER_PORT']
RANK = args.rank
WORLD_SIZE = args.world_size
MASTER = args.master
PORT = args.port
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
os.environ['MASTER'] = MASTER
os.environ['PORT'] = PORT

# print(f'{RANK=} {WORLD_SIZE=} {MASTER_ADDR=} {MASTER_PORT=}')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' if RANK==0 else '4,5,6,7'
# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
# os.environ["NCCL_SHM_DISABLE"] = "1"  
os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1" 
# os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "0" 
os.environ["NCCL_CHECKS_DISABLE"] = "1"

# torchrun --nnodes=2 --nproc-per-node=1 --rdzv-id=878 --rdzv-backend=c10d --rdzv-endpoint=33.145.114.200 dist_example.py
# python /076074/framework/example/dist_example.py --master=11.72.20.212 --port=40000 --world-size=2 --rank=0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # model_path = '/mnt/prev_nas/nanxiao/bailing'
    model_path = '/naseve/user/zhaoxin/llm_pretrain/model_hub/flood/bailing_4_10b_16k_chat_flood'

    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    MASTER = os.environ['MASTER']
    PORT = int(os.environ['PORT'])
    print(f'{MASTER=} {PORT=} {WORLD_SIZE=} {RANK=}')

    pred_path = 'tmp.jsonl'  # 推理结果文件

    """
    构造批次推理请求
    rid/input_text/output_length是必须要的
    rid: 请求id， 必须
    input_text: prompt, 必须
    output_length: 最大生成长度，够用的前提下越小越好
    """
    reqs = [Request(0,
                    input_text='<role>HUMAN</role>hello! what is your name?<role>ASSISTANT</role>',
                    output_length=1000)]
    # reqs = Reader.read_fix_dataset(model_path, max_count=1, output_length=20)
    # reqs = Reader.read_dummy_dataset(max_count=10000, input_length=100, output_length=300, flunc=0.1)

    print('start init LLM')
    worker = DistLLM(model_path,
                     n_stage=1,  # gpu count
                     n_proc=3,  # process count
                     cache_size=0.9,
                     eos_token_id=(),
                     debug=True,
                     batch_size_round_frac=0.0,  # 0.585
                     min_decode_rate=0.8,  # 0.8
                     output_file_name=pred_path,
                     output_file_mode='w+',
                     logger='example.log')
    print('finish init LLM')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    if RANK == 0:
        # do benchmark
        print(
            f'\n****************  start benchmark:{time.time() % 1000:.3f}  *******************\n')
        responses = worker.generate(reqs,
                                    input_queue,
                                    output_queues,
                                    print_count=10)
    else:
        while True:
            time.sleep(0.001)
