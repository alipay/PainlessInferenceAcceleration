
# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import time 
import os 
from vllm import LLM, SamplingParams

from flood.utils.reader import Reader

model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'

n_sample = 20000
max_output_length = 300
reqs = Reader.read_dummy_dataset(max_count=n_sample, input_length=100, output_length=max_output_length, flunc=0.1)

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, ignore_eos=True, max_tokens=max_output_length)

llm = LLM(model=model_path,gpu_memory_utilization=0.9,enforce_eager=False)

ts = time.time()
outputs = llm.generate([x.input_text for x in reqs], sampling_params)
te = time.time()
elapse = te-ts 
print(len(outputs),outputs[0])
print(f"time: {elapse:.3f}s throughput:{n_sample*max_output_length/elapse:.0f}token/s")