# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json


def preprocess_antrag(src_dir, dst_dir):
    lines = open(src_dir).readlines()
    jsons = []
    for x in lines:
        p, r = x.strip().split('[gMASK]')
        jsons.append(json.dumps({'prompt': p, 'answer': r}, ensure_ascii=False ))
    test_jsons = jsons[:1000]
    train_jsons = jsons[1000:]

    with open(dst_dir+'test.jsonl', 'w') as f:
        f.write('\n'.join(test_jsons))

    with open(dst_dir+'train.jsonl', 'w') as f:
        f.write('\n'.join(train_jsons))


def preprocess_dolly(src_dir, dst_dir, max_count=None):
    lines = open(src_dir).readlines()
    outputs = []
    jsons = [json.loads(x) for x in lines]
    for line in jsons:
        ins = line["instruction"]
        inputs = line['context']
        answer = line['response']

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

        outputs.append(json.dumps({'prompt': prompt, 'answer': answer}))

        if max_count is not None and len(outputs) >= max_count:
            break

    test_jsons = outputs[:1000]
    train_jsons = outputs[1000:]

    with open(dst_dir+'test.jsonl', 'w') as f:
        f.write('\n'.join(test_jsons))

    with open(dst_dir+'train.jsonl', 'w') as f:
        f.write('\n'.join(train_jsons))



def preprocess_gsm(src_dir, dst_dir, max_count=None):
    lines = open(src_dir).readlines()
    outputs = []
    jsons = [json.loads(x) for x in lines]
    for line in jsons:
        prompt = line["question"]
        answer = line['answer']

        outputs.append(json.dumps({'prompt': prompt, 'answer': answer}))

        if max_count is not None and len(outputs) >= max_count:
            break

    test_jsons = outputs[:1000]
    train_jsons = outputs[1000:]

    with open(dst_dir+'test.jsonl', 'w') as f:
        f.write('\n'.join(test_jsons))

    with open(dst_dir+'train.jsonl', 'w') as f:
        f.write('\n'.join(train_jsons))

def preprocess_humaneval(src_dir,dst_dir):
    train_lines = []
    test_lines = []
    for name in ['cpp','go','java','js','python']:
        filename = f'{src_dir}data_{name}_data_humaneval.jsonl'
        ls = []
        for line in open(filename):
            line = json.loads(line)
            prompt = line['prompt']
            answer = line['canonical_solution']
            ls.append(json.dumps({"prompt": prompt, "answer": answer }, ensure_ascii=False))
        length = len(ls)
        test_lines.extend(ls[:length//2])
        train_lines.extend(ls[length//2:])

    with open(f'{dst_dir}train.jsonl','w') as f:
        f.write('\n'.join(train_lines))

    with open(f'{dst_dir}test.jsonl','w') as f:
        f.write('\n'.join(test_lines))


def complete(ds='dolly_15k',model_name='llama2_7b_chat', set_name='test'):
    org_dir = f'/mntnlp/nanxiao/dataset/{ds}/{set_name}.jsonl'
    pred_dir = f'/mntnlp/nanxiao/dataset/lookahead/{ds}_{model_name}/{ds}_{model_name}.jsonl'
    kvs = {}
    for line in open(pred_dir):
        line = json.loads(line)
        kvs[line['prompt']] = (line['response'], line['ids'])
    lines = []
    hits = []
    for line in open(org_dir):
        line = json.loads(line)
        prompt = line['prompt']
        answer = line['answer']
        tup = kvs.get(prompt, None)
        if tup is None:
            hits.append(0)
            continue
        pred, ids = tup
        hits.append(1)
        d = {"prompt": prompt, "answer": answer, "pred": pred, "ids":ids}
        lines.append(json.dumps(d, ensure_ascii=False))
    print(lines[0])
    print(len(hits),sum(hits))
    with open(f'/mntnlp/nanxiao/dataset/lookahead/{ds}_{model_name}/{set_name}.jsonl','w') as f:
        f.write('\n'.join(lines))

def rename(src_dir, dst_dir, names=None):
    jsons = []
    for line in open(src_dir):
        line = json.loads(line)
        d = {}
        for k,v in names.items():
            d[v] = line[k]
        jsons.append(json.dumps(d, ensure_ascii=False))
    with open(dst_dir,'w') as f:
        f.write('\n'.join(jsons))

def count_tokens(dataset_dir, model_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt_tokens = []
    answer_tokens = []
    for line in open(dataset_dir):
        line = json.loads(line)
        pt = len(tokenizer(line['prompt']).input_ids)
        at = len(tokenizer(line['answer']).input_ids)
        prompt_tokens.append(pt)
        answer_tokens.append(at)
    print(f'size:{len(prompt_tokens)} prompt:{sum(prompt_tokens)/len(prompt_tokens)} answer:{sum(answer_tokens)/len(answer_tokens)}')


# src_dir = '/mntnlp/nanxiao/dataset/antrag_8k/antrag.txt'
# dst_dir = '/mntnlp/nanxiao/dataset/antrag_8k/'
# preprocess_antrag(src_dir, dst_dir)
                
# src_dir = '/mntnlp/nanxiao/dataset/dolly_15k/databricks-dolly-15k.jsonl'
# dst_dir = '/mntnlp/nanxiao/dataset/dolly_15k/'
# preprocess_dolly(src_dir, dst_dir)

# src_dir = '/mntnlp/nanxiao/dataset/gsm_8k/gsm_8k.jsonl'
# dst_dir = '/mntnlp/nanxiao/dataset/gsm_8k/'
# preprocess_gsm(src_dir, dst_dir)

# src_dir = '/mntnlp/nanxiao/dataset/humaneval-x/'
# dst_dir = '/mntnlp/nanxiao/dataset/humaneval-x/'
# split_humaneval(src_dir, dst_dir)

# src_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/train.jsonl'
# dst_dir = '/mntnlp/nanxiao/dataset/lookahead/antrag_8k_antglm_10b/train.jsonl'
# names = {'prompt': 'prompt', 'response': 'answer', 'pred': 'pred', 'ids': 'ids'}
# rename(src_dir, dst_dir, names=names)


# complete(ds='dolly_15k', model_name='chatglm2', set_name='train')

# dataset_dir = '/mntnlp/nanxiao/dataset/gsm_8k/train.jsonl'
# model_dir = '/mntnlp/common_base_model/llama2-7b-chat'
# count_tokens(dataset_dir,model_dir)