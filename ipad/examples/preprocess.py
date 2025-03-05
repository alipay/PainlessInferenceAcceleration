# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import json

def preporcess_dolly():
    src_dir = '/Users/yaozhao/dataset/dataset/databricks-dolly-15k.jsonl'
    lines = []
    for line in open(src_dir):
        line = json.loads(line)
        lines.append(json.dumps({'prompt': line['instruction'] + line['context'] ,'response': line['response'] }))
        if len(lines) > 1000:
            break
    dst_dir = '/Users/yaozhao/dataset/dataset/dolly_15k.jsonl'
    with open(dst_dir,'w+') as f:
        f.write('\n'.join(lines))


preporcess_dolly()