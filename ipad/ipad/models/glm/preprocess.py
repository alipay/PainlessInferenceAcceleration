# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import json
import sys
import csv


# sys.path.append('/mntnlp/common_base_model/antglm_rotary_100k/')
sys.path.insert(0,'/mntnlp/common_base_model/glm-10b-sft-2k')
from tokenization_glm import GLMChineseTokenizer

# tokenizer = GLMChineseTokenizer.from_pretrained('/mntnlp/luohe/exp/searchx/multidoc/antglm_100k')
tokenizer = GLMChineseTokenizer.from_pretrained('/mntnlp/common_base_model/glm-10b-sft-2k')


# input_name = '/mntnlp/nanxiao/searchx_ext/apm_searchx_high_quality_query_7k_paragraph_extract_train_data.csv'
# output_name = '/mntnlp/nanxiao/searchx_ext/train_64k_short.jsonl'

input_name = '/mntnlp/nanxiao/searchx_ext/apm_searchx_high_quality_paragraph_extract_test.csv'
output_name = '/mntnlp/nanxiao/searchx_ext/test_2k_short.jsonl'

# lines = open(input_name,'r').readlines()
lines = csv.reader(open(input_name))

max_input_len = 2048 - 256
jsons = []
dedups = set()
lengths = [0]*100
for i, line in enumerate(lines):
    if i == 0:
        continue
    # line = json.loads(line)
    # ins = line['prompt']
    ins = line[0]
    res = line[1]
    # pred = None

    # ins = ins[len('[Round 0]\n\n问：'):-len('\n【答案】：\n\n答：')]

    qid = tokenizer(ins).input_ids
    if i<10:
        print(qid[-10:])
    lengths[len(qid)//100] += 1
    if len(qid) > max_input_len:
        # qid = qid[:max_input_len-5] + qid[-5:]
        # ins = tokenizer.decode(qid)
        continue

    if ins in dedups:
        continue
    else:
        dedups.add(ins)
    # jsons.append(json.dumps({'prompt':ins, 'response': res ,'predict': pred}, ensure_ascii=False))
    if len(jsons) >= 400:
        break
    jsons.append(json.dumps({'prompt':ins, 'response': res }, ensure_ascii=False))
    # jsons.append(json.dumps({'prompt':ins, 'response':line['response'].split('<|endoftext|><|startofpiece|>')[1].replace('<|endofpiece|>','').strip() }, ensure_ascii=False))

print([(i*100, x) for i,x in enumerate(lengths)])

with open(output_name , 'w') as f:
    f.write('\n'.join(jsons))

print(f'{len(jsons)=}')
print('done!')
