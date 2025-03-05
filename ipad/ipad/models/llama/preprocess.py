# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json
import csv
import numpy as np


from transformers import LlamaTokenizer
from transformers import GPT2TokenizerFast

llama_tokenizer = LlamaTokenizer.from_pretrained('/mntnlp/common_base_model/llama2-7b-chat')
opt_tokenizer = GPT2TokenizerFast.from_pretrained('/mntnlp/common_base_model/opt_6b7')
input_name = '/mntnlp/nanxiao/dataset/databricks-dolly-15k.jsonl'
output_name = '/mntnlp/nanxiao/dataset/dolly_15k.jsonl'
# input_name = '/mntnlp/nanxiao/dataset/alpaca_data_cleaned.json'
# output_name = '/mntnlp/nanxiao/dataset/alpaca_50k.jsonl'
lines = open(input_name).readlines()

lines = [json.loads(line) for line in lines]
# lines = json.loads('\n'.join(lines))

jsons = []
indices = []
for i, line in enumerate(lines):

    # dolly
    ins = line["instruction"]
    doc = line["context"]
    res = line['response']
    cat = line['category']

    # ins = line["instruction"]
    # doc = line["input"]
    # res = line['output']
    # cat = 'default'

    if len(doc) == 0:
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
        prompt = template.format(instruction=ins, input=doc)

    prompt = prompt.replace('\n', '')
    if len(llama_tokenizer(prompt, add_special_tokens=False).input_ids) > 256:
        continue

    if len(opt_tokenizer(prompt, add_special_tokens=False).input_ids) > 256:
        continue

    jsons.append(json.dumps({'prompt':prompt, 'response': res, 'cat': cat }, ensure_ascii=False))
    indices.append(i)

print(f'size:{len(lines)}')

with open(output_name,'w') as f:
    f.write('\n'.join(jsons))

print('done!')
