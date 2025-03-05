# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json
import csv


n = 0.0
n_t = 0.0
qs = []
ans = []
preds = []
lines = []
for i,line in enumerate(open('/mntnlp/nanxiao/guix/train.csv','r')):
    if i == 0:
        continue
    ps = line.strip().strip('"').split('",')
    q = ps[0].strip().strip('"')
    a = ps[1].strip().strip('"')
    p = ps[1].strip().strip('"')
    qs.append(q)
    ans.append(a)
    preds.append(p)
    lines.append(q + '\t\t\t\t' + a + '\t\t\t\t' + p)
print(f'sample:{len(qs)}')

# with open('/mntnlp/nanxiao/guix/test.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for i, row in enumerate(spamreader):
#         if i > 5:
#             break
#         for c in row:
#             print(c)  

with open('/mntnlp/nanxiao/guix/qas_2k.txt','w') as f:
    f.write('\n'.join(lines))
