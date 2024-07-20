# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import numpy as np
import pickle
import sys
import json



# sys.path.append('..')
# from antglm.tokenization_glm import GLMChineseTokenizer

# token_dir = '/mntnlp/nanxiao/antgmm'
# tokenizer = GLMChineseTokenizer.from_pretrained(token_dir)

# # input_names = ['/mnt_alipayshnas/workspace/lekun.ljj/LMM/data/adver/distill_data/dataFile.txt']
# # label_names = ['/mnt_alipayshnas/workspace/lekun.ljj/LMM/data/adver/distill_data/test_res.txt']
# # input_names = ['/mntnlp/nanxiao/blip/blip_test_data.txt']
# # label_names = ['/mntnlp/nanxiao/blip/blip_test_label.txt']
# input_names = ['/mntnlp/nanxiao/blip/blip_test_data_3k.txt']
# label_names = ['/mntnlp/nanxiao/blip/blip_test_label_3k.txt']
# ns = [3000]
# queries = []
# answers = []
# preds = []
# embs = np.zeros((sum(ns),32,3072),dtype=np.float16)
# qas = set()
# lens = [0, 1600]  
# # lens = [17, 512]
# nc = 0
# for it in range(1):
#     fr = open(input_names[it], 'rb')
#     fl = open(label_names[it],'r')
#     fl.readline()
#     for i in range(ns[it]):
#         try:
#             data = pickle.load(fr)
#             label_text = fl.readline()
#         except Exception as e:
#             print(e)
#             break

#         token_ids = list(data['outputs'][0])
#         idx = token_ids.index(50006)
#         input_ids = token_ids[:idx-2]
#         input_text = tokenizer.decode(input_ids[33:])
#         # if 'http:' in input_text:
#         #     input_text = input_text.split('http:')[0]
#         query = '[CLS]'*32+input_text+'[gMASK]'

#         output_ids = token_ids[idx+1:]

#         answer = label_text.split('\t')[3].strip()

#         pred = tokenizer.decode(output_ids).replace('<|endofpiece|>','')
#         qa = query + answer
#         # if "'from': 'human'" in input_text or qa in qas:
#         #     print(qa)
#             # continue

#         if i <= 3:
#             print(query, answer, pred)

#         if len(output_ids) < lens[0] or len(output_ids) > lens[1]:
#             continue

#         if 'Observe the image carefully' in query:
#             continue

#         queries.append(query)
#         answers.append(answer)
#         preds.append(pred)
#         qas.add(qa)

#         emb = data['query_embeds'][0]
#         embs[nc] = emb

#         nc += 1
#         if i % 100 == 0:
#             print(f'epoch:{it} iter:{i} count:{nc}')

#         # print(f'QA size:{len(token_ids)} ids:{token_ids}')    
#         # print(f'Q size:{len(queries[-1])} ids:{input_ids}')    
#         # print(f'A size:{len(answers[-1])} ids:{output_ids}')    
#         # print('Q:'+queries[-1])    
#         # print('A:'+answers[-1])


#         # inputs = tokenizer(queries[-1], padding="max_length", max_length=128,return_tensors="pt",truncation=True)
#         # inputs = tokenizer.build_inputs_for_generation(inputs,targets=answers[-1],max_gen_length=128,padding=True)
#         # attention_mask = inputs['attention_mask'].bfloat16().cuda()
#         # labels = inputs['labels'].cuda()
#         # input_ids = inputs['input_ids'].cuda()
#         # print('input_ids:', input_ids)
#         # position_ids = inputs['position_ids'].cuda()
#         # labels = inputs['labels'].cuda()

# embs = embs[:nc]
# print(f'length in {lens} samples:{nc}')

# # total samples:61337

# with open(token_dir+'/test_qas_3k.txt','w') as f:
#     qas = []
#     for i, q in enumerate(queries):
#         line = json.dumps({'prompt':q,'response':answers[i], 'pred':preds[i]})
#         qas.append(line)
#     f.write('\n'.join(qas))
# print(token_dir+'/test_qas_3k.txt')

# with open(token_dir+'/test_embs_3k.bin', 'wb') as f:
#     np.save(f, embs)
# print(token_dir+'/test_embs_3k.bin')


# def convert_format(src_dir, dst_dir):
#     jsons = []
#     for line in open(src_dir,'r'):
#         q,a,p = line.strip().split('\t\t\t\t')
#         jsons.append(json.dumps({'prompt':q, 'response': a, 'pred':p}))
#     with open(dst_dir,'w') as f:
#         f.write('\n'.join(jsons))

# convert_format('/mntnlp/nanxiao/blip/qas_100k.txt','/mntnlp/nanxiao/antgmm/qas_100k.txt')
# convert_format('/mntnlp/nanxiao/blip/test_qas_500.txt','/mntnlp/nanxiao/antgmm/test_qas_500.txt')





sample_dir = '/mntnlp/nanxiao/antgmm/qas_100k.txt'
emb_dir = '/mntnlp/nanxiao/blip/embs_100k.bin'
logit_dir = '/mntnlp/nanxiao/blip/logits_100k.npz'

indices = open('/mntnlp/nanxiao/blip/sample_indices.json','r').readline()
indices = json.loads(indices)
print(f'size:{len(indices)}')

lines = []
for i, line in enumerate(open(sample_dir,'r')):
    lines.append(line.strip())

lines = [lines[i] for i in indices]
print(lines[:2])
with open('/mntnlp/nanxiao/antgmm/qas_16k.txt', 'w') as f:
    f.write('\n'.join(lines))

# with open(emb_dir, 'rb') as f:
#     embeddings = np.load(f)

# with open('/mntnlp/nanxiao/blip/embs_16k.bin', 'wb') as f:
#     np.save(f, embeddings[indices])

# with open(logit_dir, 'rb') as f:
#     caches = np.load(f)
#     logit_cache = caches['logits']

# with open('/mntnlp/nanxiao/blip/logits_16k.npz', 'wb') as f:
#     np.savez(f, logits=logit_cache[indices])
