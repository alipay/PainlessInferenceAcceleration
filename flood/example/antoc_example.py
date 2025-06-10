# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import json
import os
import random
import time
import argparse

import torch.multiprocessing as mp
from transformers import AutoTokenizer
from flood.facade.llm import LLM
from flood.utils.reader import Reader
from flood.utils.request import Request
from flood.facade.dist_llm import DistLLM

random.seed(7)

class AntocUtil():
    @staticmethod
    def read_antoc_dataset(path_name, tokenizer_path, task_names=None,
                           max_count=1000000):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        file_names = Reader.list_files(path_name, postfix='json')
        # def fn(s):
        #     pn = s[0].split('.')[0]
        #     pns = pn.split('_')
        #     p1, p2 = '_'.join(pns[:-1]), str(1000+int(pns[-1]))
        #     return p1+'_'+p2
        # file_names = sorted(file_names, key=fn)
        reqs = []
        for filename, path_name in file_names:
            hit = True
            if task_names is not None:
                hit = False
                for tn in task_names:
                    if filename.startswith(tn):
                        hit = True
                        break
            if not hit:
                continue
            if len(reqs) >= max_count:
                break
            lines = []
            for line in open(path_name, 'r'):
                lines.append(line.strip('\n'))
            lines = json.loads('\n'.join(lines))
            task_type = None
            for key, value in sorted(lines.items(), key=lambda x: int(x[0])):
                prediction = value['prediction']
                rid = filename + ':' + key
                if isinstance(prediction, str):  # ppl tasks
                    full_prompt = value['label: A']['prompt']
                    input_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
                    length = len(tokenizer.decode(input_ids[-1:]))
                    prompt = full_prompt[:-length]
                    target_tokens = []
                    target_ids = []
                    for k, v in value.items():
                        if k.startswith('label: '):
                            target_tokens.append(k[7])
                            target_id = tokenizer(v['prompt'][-length:], add_special_tokens=False).input_ids
                            assert len(target_id) == 1
                            target_ids.append(target_id)
                    input_length = len(input_ids)
                    req = Request(rid, input_text=prompt, input_length=input_length, output_length=1, content=None,
                                  target_tokens=target_tokens, target_ids=target_ids)
                elif isinstance(prediction, int):  # muti-tokens ppl tasks
                    task_type = 'ppl'
                    full_input_ids = []
                    for k, v in value.items():
                        if k.startswith('label: '):
                            full_input_ids.append(tokenizer(v['prompt'], add_special_tokens=False).input_ids)
                    common_length = 0
                    for j in range(len(full_input_ids[0])):
                        if all(full_input_ids[0][j] == full_input_ids[i][j] for i in range(1, len(full_input_ids))):
                            common_length += 1
                        else:
                            break
                    length = len(tokenizer.decode(full_input_ids[0][:common_length]))
                    prompt = value['label: 0']['prompt'][:length]

                    assert all(full_input_ids[0][:common_length] == full_input_ids[i][:common_length] for i in range(1, len(full_input_ids)))
                    assert any(full_input_ids[0][common_length] != full_input_ids[i][common_length] for i in range(1, len(full_input_ids)))

                    target_tokens = []
                    target_ids = []
                    max_output_length = 0
                    for k, v in value.items():
                        if k.startswith('label: '):
                            target_tokens.append(k[7])
                            target_id = tokenizer(v['prompt'][length:], add_special_tokens=False).input_ids
                            if len(target_id) > max_output_length:
                                max_output_length = len(target_id)
                            target_ids.append(target_id)
                    input_length = common_length
                    req = Request(rid, input_text=prompt, input_length=input_length, output_length=max_output_length, content=None,
                                  target_tokens=target_tokens, target_ids=target_ids)
                else:  # gen tasks
                    task_type = 'gen'
                    assert prediction['task_type'] == 'gen'
                    prompt = prediction.get('format_prompt') or prediction.get(
                        'prompt')
                    output_length = prediction.get(
                        'max_length') or prediction.get('max_gen_len')
                    input_length = len(tokenizer(prompt).input_ids)
                    req = Request(rid, input_text=prompt,
                                  input_length=input_length,
                                  output_length=output_length, content=value)
                reqs.append(req)
                if len(reqs) >= max_count:
                    break
            print(
                f'file_name:{filename} index:{len(reqs)} task_type:{task_type}')
        Reader.stat(reqs)
        return reqs
    @staticmethod
    def replace_antoc_prediction(src_dir, pred_dir, dst_dir, task_names=None):
        for filename, filepath in Reader.list_files(dst_dir, postfix='json'):
            hit = True
            if task_names is not None:
                hit = False
                for tn in task_names:
                    if filename.startswith(tn):
                        hit = True
                        break
            if not hit:
                continue
            os.remove(filepath)
        os.makedirs(dst_dir, exist_ok=True)
        predictions = {}
        for line in open(pred_dir):
            line = line.strip()
            if len(line) == 0:
                continue
            line = json.loads(line)
            rid = line['rid']
            prediction = line['output_text']
            predictions[rid] = prediction
        for filename, filepath in Reader.list_files(src_dir, postfix='json'):
            hit = True
            if task_names is not None:
                hit = False
                for tn in task_names:
                    if filename.startswith(tn):
                        hit = True
                        break
            if not hit:
                continue
            with open(filepath) as f:
                lines = json.load(f)
            keys = list(lines.keys())
            count = 0
            for key in keys:
                sample = lines[key]
                rid = filename + ':' + key
                if rid in predictions:
                    prediction = predictions[rid]
                    if isinstance(sample['prediction'], str) or isinstance(sample['prediction'], int):  # ppl 
                        scores = {x.split(':')[0]: float(x.split(':')[1]) for x
                                  in prediction.split(' ')}
                        ls = []
                        for k, v in sample.items():
                            if k.startswith('label:'):
                                label = k[7]
                                score = scores[label]
                                ls.append((label, score))
                                sample[f'label: {label}']['PPL'] = score
                        ls = sorted(ls, key=lambda x: x[1], reverse=False)
                        if isinstance(sample['prediction'], str):
                            sample['prediction'] = ls[0][0]
                        elif isinstance(sample['prediction'], int):
                            sample['prediction'] = int(ls[0][0])
                    else:
                        sample['prediction'] = prediction
                    count += 1
            print(f'replace {count}/{len(lines)} predictions in {filename}')
            with open(os.path.join(dst_dir, filename), 'w+') as f:
                f.write(json.dumps(lines, indent=4, ensure_ascii=False))
    @staticmethod
    def extract_antoc_request(src_dir, pred_dir, tokenizer_path,
                              task_names=None):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        predictions = {}
        if os.path.exists(pred_dir):
            for line in open(pred_dir):
                line = line.strip()
                if len(line) == 0:
                    continue
                line = json.loads(line)
                rid = line['rid']
                prediction = line['output_text']
                predictions[rid] = prediction
        reqs = []
        for filename, filepath in Reader.list_files(src_dir, postfix='json'):
            hit = True
            if task_names is not None:
                hit = False
                for tn in task_names:
                    if filename.startswith(tn):
                        hit = True
                        break
            if not hit:
                continue
            with open(filepath) as f:
                lines = json.load(f)
            fn = filename
            keys = list(lines.keys())
            count = 0
            for key in keys:
                rid = fn + ':' + key
                if rid in predictions:
                    continue
                count += 1
                value = lines[key]
                prediction = value['prediction']
                if isinstance(prediction, str):  # ppl tasks
                    full_prompt = value['label: A']['prompt']
                    input_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
                    length = len(tokenizer.decode(input_ids[-1:]))
                    prompt = full_prompt[:-length]
                    target_tokens = []
                    target_ids = []
                    for k, v in value.items():
                        if k.startswith('label: '):
                            target_tokens.append(k[7])
                            target_id = tokenizer(v['prompt'][-length:], add_special_tokens=False).input_ids
                            assert len(target_id) == 1
                            target_ids.append(target_id)
                    input_length = len(input_ids)
                    req = Request(rid, input_text=prompt, input_length=input_length, output_length=1, content=None,
                                  target_tokens=target_tokens, target_ids=target_ids)
                elif isinstance(prediction, int):  # muti-tokens ppl tasks
                    task_type = 'ppl'
                    full_input_ids = []
                    for k, v in value.items():
                        if k.startswith('label: '):
                            full_input_ids.append(tokenizer(v['prompt'], add_special_tokens=False).input_ids)
                    common_length = 0
                    for j in range(len(full_input_ids[0])):
                        if all(full_input_ids[0][j] == full_input_ids[i][j] for i in range(1, len(full_input_ids))):
                            common_length += 1
                        else:
                            break
                    length = len(tokenizer.decode(full_input_ids[0][:common_length]))
                    prompt = value['label: 0']['prompt'][:length]

                    assert all(full_input_ids[0][:common_length] == full_input_ids[i][:common_length] for i in range(1, len(full_input_ids)))
                    assert any(full_input_ids[0][common_length] != full_input_ids[i][common_length] for i in range(1, len(full_input_ids)))

                    target_tokens = []
                    target_ids = []
                    max_output_length = 0
                    for k, v in value.items():
                        if k.startswith('label: '):
                            target_tokens.append(k[7])
                            target_id = tokenizer(v['prompt'][length:], add_special_tokens=False).input_ids
                            if len(target_id) > max_output_length:
                                max_output_length = len(target_id)
                            target_ids.append(target_id)
                    input_length = common_length
                    req = Request(rid, input_text=prompt, input_length=input_length, output_length=max_output_length, content=None,
                                  target_tokens=target_tokens, target_ids=target_ids)
                else:  # gen tasks
                    assert prediction['task_type'] == 'gen'
                    prompt = prediction.get('format_prompt') or prediction.get(
                        'prompt')
                    output_length = prediction.get(
                        'max_length') or prediction.get('max_gen_len')
                    input_length = len(tokenizer(prompt).input_ids)
                    req = Request(rid, input_text=prompt,
                                  input_length=input_length,
                                  output_length=output_length, content=value)
                reqs.append(req)
            print(f'find {count}/{len(keys)} unprocessed samples in {filename}')
        Reader.stat(reqs)
        return reqs


parser = argparse.ArgumentParser()
parser.add_argument("--rank", "--rank", type=int, default=0)
parser.add_argument("--world-size", "--world-size", type=int, default=1)
parser.add_argument("--master", "--master", type=str, default='127.0.0.1')
parser.add_argument("--port", "--port", type=str, default='40000')
args = parser.parse_args()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
RANK = args.rank
WORLD_SIZE = args.world_size
MASTER = args.master
PORT = args.port
os.environ['FLOOD_RANK'] = str(RANK)
os.environ['FLOOD_WORLD_SIZE'] = str(WORLD_SIZE)
os.environ['FLOOD_MASTER'] = MASTER
os.environ['FLOOD_PORT'] = PORT

# print(f'{RANK=} {WORLD_SIZE=} {MASTER_ADDR=} {MASTER_PORT=}')

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_SHM_DISABLE"] = "1"  
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_CHECKS_DISABLE"] = "1"
# os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1" 
# os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "0" 

# USAGE
# single-node
# python antoc_example.py
# multi-node
# on node 1:  python /076074/framework/example/antoc_example.py
# --master=ip --port=40000 --world-size=2 --rank=0
# on node 2:  python /076074/framework/example/antoc_example.py
# --master=ip --port=40000 --world-size=2 --rank=1

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    WORLD_SIZE = int(os.environ['FLOOD_WORLD_SIZE'])
    RANK = int(os.environ['FLOOD_RANK'])

    model_path = '/mnt/nas_acr89/jingyue/ling-moe-lite-chat'
    data_path = '/mnt/nas_acr89/jingyue/ling-moe-lite-chat-ppl/predictions/antglm-bt1-uni-742824ec'
    pred_path = 'tmp.jsonl'
    
    replace_path = '/ossfs/workspace/antoc/tmp_antoc_result/Qwen__Qwen2.5-0.5B-Instruct/predictions/antglm-bt1-uni-d4fff1a6'
    reuse = False
    # read prompt
    task_names = ['piqa', 'ARC-c', 'ARC-e', 'hellaswag',]

    if reuse:
        reqs = AntocUtil.extract_antoc_request(data_path, pred_path, model_path,
                                               task_names=task_names)
    else:
        reqs = AntocUtil.read_antoc_dataset(data_path, model_path,
                                            task_names=task_names,
                                            max_count=1000000)
    if len(reqs) == 0:
        print('no samples!')
        exit()
    if WORLD_SIZE == 1:
        worker = LLM(model_path,
                        #  cache_dtype=torch.bfloat16,
                        n_stage=2,  # gpu count
                        n_proc=3,  # process count
                        chunk_size=1024,
                        #  model_dtype=torch.float8_e4m3fn,
                        # max_concurrency=1024,
                        # cache_size=16000,
                        slot_fully_alloc_under=10240,
                        tune_alloc_size=False,
                        eos_token_id=None,
                        debug=False,
                        kernels=('sa',),
                        logger='antoc.log')
    elif WORLD_SIZE == 2:
        worker = DistLLM(model_path,
                        #  cache_dtype=torch.bfloat16,
                        n_stage=2,  # gpu count
                        n_proc=3,  # process count
                        chunk_size=1024,
                        #  model_dtype=torch.float8_e4m3fn,
                        # max_concurrency=1024,
                        # cache_size=16000,
                        slot_fully_alloc_under=10240,
                        tune_alloc_size=False,
                        eos_token_id=None,
                        debug=False,
                        kernels=('sa',),
                        logger='antoc.log')
    else:
        raise ValueError(f'unsupported world size: {WORLD_SIZE}')
    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)
    # do benchmark
    print(
        f'\n****************  start benchmark:{time.time() % 1000:.3f}  *******************\n')
    if RANK == 0:
        output_file_mode='a+' if reuse else 'w+'
        handler = open(pred_path, output_file_mode)
        for i, req in enumerate(worker.request_stream_generate(reqs,
                                                input_queue,
                                                output_queues,
                                                print_count=0)):
            kvs = req.__dict__
            dumps = {}
            keys = ('rid', 'output_text', 'input_ids')
            for k, v in kvs.items():
                if k in keys:
                    dumps[k] = v
            handler.write(json.dumps(dumps, ensure_ascii=False) + '\n')
            if i <= 3:
                print('\n\n')
                print(f'prompt-{i}: ', req.input_text)
                print(f'answer-{i}: ', req.output_text)
                print(f'target_tokens-{i}: ', req.target_tokens)
                print(f'target_ids-{i}: ', req.target_ids)
        handler.close()
        # replace prompt
        AntocUtil.replace_antoc_prediction(data_path, pred_path, replace_path,
                                        task_names=task_names)
    else:
        while True:
            time.sleep(0.001)