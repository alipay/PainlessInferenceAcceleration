# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json
import os
import random
import time

import torch.multiprocessing as mp

from flood.common.llm import LLM
from flood.utils.reader import Reader
from flood.utils.request import Request

random.seed(7)


class AntocUtil():

    @staticmethod
    def read_antoc_dataset(path_name, tokenizer_path, task_names=None,
                           max_count=1000000):
        tokenizer = Reader.load_tokenizer(tokenizer_path)
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
                    task_type = 'ppl'

                    full_prompt = value['label: A']['prompt']
                    input_ids = tokenizer(full_prompt,
                                          add_special_tokens=False).input_ids
                    length = len(tokenizer.decode(input_ids[-1:]))
                    prompt = full_prompt[:-length]

                    target_tokens = []
                    target_ids = []
                    for k, v in value.items():
                        if k.startswith('label: '):
                            target_tokens.append(k[7])
                            target_id = tokenizer(v['prompt'][-length:],
                                                  add_special_tokens=False).input_ids
                            assert len(target_id) == 1
                            target_ids.append(target_id[0])
                    print(f'{target_tokens=} {target_ids=}')
                    input_length = len(tokenizer(prompt).input_ids)
                    req = Request(rid, input_text=prompt,
                                  input_length=input_length, output_length=1,
                                  content=None, target_tokens=target_tokens,
                                  target_ids=target_ids)
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
                    if isinstance(sample['prediction'], str):  # ppl 
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
                        sample['prediction'] = ls[0][0]
                    else:
                        sample['prediction'] = prediction
                    count += 1
            print(f'replace {count}/{len(lines)} predictions in {filename}')
            with open(os.path.join(dst_dir, filename), 'w+') as f:
                f.write(json.dumps(lines, indent=4, ensure_ascii=False))

    @staticmethod
    def extract_antoc_request(src_dir, pred_dir, tokenizer_path,
                              task_names=None):

        tokenizer = Reader.load_tokenizer(tokenizer_path)

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
                    prompt = value['label: A']['prompt'][:-1]
                    target_tokens = []
                    for k, v in value.items():
                        if k.startswith('label:'):
                            label = k[7]
                            target_tokens.append(label)
                    input_length = len(tokenizer(prompt).input_ids)
                    rid = filename.split('.')[0] + ':' + key
                    req = Request(rid, input_text=prompt,
                                  input_length=input_length, output_length=1,
                                  content=None, target_tokens=target_tokens)
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


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # model_path = '/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct'  # llama
    # model_path = '/mntnlp/common_base_model/Llama-3.1-8B-Instruct' # llama
    # model_path = '/mnt/nas_acr89/nanxiao/chat_80b'  # bailing
    # model_path = '/naseve/user/zhaoxin/llm_pretrain/model_hub/Qwen2.5-7B'
    # model_path = '/mnt/nas_acr89/jingyue/20b_base_flood'
    # model_path = '/mnt/nas_acr89/jingyue/Bailing-4-20B-4K-base-20241029' 
    model_path = '/mnt/prev_nas/chatgpt/release/1029/20b/base/Bailing-4.0-20B-4K-base-20241029'
    # model_path = '/mnt/nas_acr89/jingyue/10b_base_flood'

    # data_path = '/mntnlp/nanxiao/dataset/ocl/outputs/workdir/predictions/llama-3-8b-instruct-hf-flood-b239fb44-process' # cmmlu & gsm8k
    # data_path = '/mntnlp/nanxiao/dataset/antoc/bbh_80b_chat_0920'  # bbh
    # data_path = '/mntnlp/nanxiao/dataset/oc/opencopmass/workdir/default/predictions/antglm-flood-c4085f10' # cmmlu
    # data_path = '/ossfs/workspace/tmp/outputs/workdir/predictions/antglm-flood-2fef8032-process'  # qwen full
    # data_path = '/mntnlp/nanxiao/dataset/mix_gen_ppl_task'  # samples from ppl&gen
    data_path = '/mntnlp/jingyue/datasets/flood_72500083/workdir/predictions/antglm-flood-dd0fc0ec-process'

    pred_path = 'tmp.jsonl'

    # replace_path = '/ossfs/workspace/antoc/tmp/flood/predictions/chat_80b_hf'
    replace_path = '/ossfs/workspace/antoc/tmp/flood/predictions/Meta-Llama-3-8B-Instruct_hf'
    # replace_path = '/ossfs/workspace/tmp/input_replace'

    reuse = False

    # read prompt
    # task_names = ['agieval','bbh','ceval', 'cmmlu','gsm8k','hellaswag', 'lukaemon_mmlu', 'mbpp', 'math','openai_humaneval','triviaqa']
    # task_names = ['lukaemon_mmlu_abstract_algebra','agieval']
    task_names = ['GPQA']  # ARC_c_ppl_2ef631
    if reuse:
        reqs = AntocUtil.extract_antoc_request(data_path, pred_path, model_path,
                                               task_names=task_names)
    else:
        reqs = AntocUtil.read_antoc_dataset(data_path, model_path,
                                            task_names=task_names,
                                            max_count=100000)
    if len(reqs) == 0:
        print('no samples!')
        exit()

    worker = LLM(model_path,
                 n_stage=1,  # gpus
                 n_proc=1,
                 eos_token_id=None,  # llama:(128001,128008,128009)
                 debug=True,
                 output_file_name=pred_path,
                 output_file_mode='a+' if reuse else 'w+',
                 output_field_names=('output_text', 'input_ids'),
                 logger='antoc.log')

    # start process
    input_queue, chunk_queue, working_queue, output_queues = worker.initialize()
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)

    # do benchmark
    print(
        f'\n****************  start benchmark:{time.time() % 1000:.3f}  *******************\n')
    worker.generate(reqs, input_queue, output_queues, print_count=10)

    # replace prompt
    AntocUtil.replace_antoc_prediction(data_path, pred_path, replace_path,
                                       task_names=task_names)
