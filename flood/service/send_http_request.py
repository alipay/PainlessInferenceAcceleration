# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import requests

"""
curl -X POST -N -d '{"prompt": "What is the capital of France?", "max_length":200}' http://localhost:30010/generate
"""

url = "http://localhost:30010/generate"
data = {"prompt": "hi" * 60, "max_length": 100}

for response in requests.post(url, json=data, stream=True):
    text = response.decode('utf-8').replace('\n\n', '')
    print(text, flush=True, end='')


# with requests.post(url, json=data, stream=True) as r:
#     for chunk in r.iter_content(1):
#         print(chunk, flush=True, end='')


# import httpx
# with httpx.stream('POST', url, json=data, timeout=120) as r:
#     for chunk in r.iter_text():
#         print(chunk, flush=True, end='')
