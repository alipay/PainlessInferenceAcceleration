# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json


def preprocess_txt(src_dir, dst_dir,ds='search'):
    lines = open(src_dir).readlines()
    prompts = [x.strip().split('[gMASK]')[0] for x in lines]
    jsons = []
    for p in prompts:
        jsons.append(json.dumps({'prompt':q}))
    with open(dst_dir,'w') as f:
        f.write('\n'.join(jsons))

def preprocess_json(src_dir, dst_dir):
    prompts = []
    for d in json.loads('\n'.join(lines)):
        prompts.append(d['instruction']+' '+ d['input'])
        if max_count is not None and len(prompts) >= max_count:
            break
    jsons = []
    for p in prompts:
        jsons.append(json.dumps({'prompt':q}))
    with open(dst_dir,'w') as f:
        f.write('\n'.join(jsons))

def preprocess_jsonl(src_dir, dst_dir,ds='coig',max_count=None):
    lines = open(src_dir).readlines()
    prompts = []
    for d in lines:
        d = json.loads(d)
        prompts.append(d['prompt'])
        if max_count is not None and len(prompts) >= max_count:
            break

    jsons = []
    for p in prompts:
        jsons.append(json.dumps({'prompt':q}))
    with open(dst_dir,'w') as f:
        f.write('\n'.join(jsons))

def preprocess_csv(src_dir, dst_dir, ds=None, max_count=None):
    import csv
    lines = csv.reader(open(src_dir))

    jsons = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        p = line[2]
        jsons.append(json.dumps({'prompt':p}))
    with open(dst_dir,'w') as f:
        f.write('\n'.join(jsons))

def preprocess_dolly(src_dir, dst_dir, ds='dolly',max_count=None):
    lines = open(filename).readlines()
    prompts = []
    if ds == 'alpaca':
        jsons = json.loads('\n'.join(lines))
    else:
        jsons = [json.loads(x) for x in lines]
    for line in jsons:
        ins = line["instruction"]
        inputs = line['context'] if ds=='dolly' else line['input']
        answer = line['response'] if ds=='dolly' else line['output']

        if len(inputs) == 0:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            prompt = template.format(instruction=ins)
        else:
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )
            prompt = template.format(instruction=ins, input=inputs)

        prompt = prompt.replace('\n', '')
        if len(prompt) >= 4096:
            continue
        prompts.append(prompt)

        if max_count is not None and len(prompts) >= max_count:
            break

    jsons = []
    for p in prompts:
        jsons.append(json.dumps({'prompt':q}))
    with open(dst_dir,'w') as f:
        f.write('\n'.join(jsons))
