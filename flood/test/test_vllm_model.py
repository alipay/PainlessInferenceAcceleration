# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams


model_name = "/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = LLM(model_name, quantization=None, tensor_parallel_size=1,
            enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=1024,
            max_seq_len_to_capture=1024)

prompts = ['hello! what is your name?',
           'tell me a joke!',
           '中国的首都是哪里？',
           '杭州在哪里？']

for i, prompt in enumerate(prompts):
    messages = [
        {"role": "system",
         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sample_params = SamplingParams(max_tokens=128, ignore_eos=False,
                                   skip_special_tokens=False)
    result = model.generate(text, sampling_params=sample_params)
    response = result[0].outputs[0].text
    print(f'\nprompt-{i}:', text.replace('\n', '\\n'))
    print(f'response-{i}:', response.replace('\n', '\\n') + '\n')
