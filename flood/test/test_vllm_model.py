# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams


model_name = "/agent/nanxiao/models/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = LLM(
    model_name,
    quantization=None,
    tensor_parallel_size=1,
    enforce_eager=False,
    gpu_memory_utilization=0.90,
    max_model_len=10240,
    max_seq_len_to_capture=1024,
)

# prompts = ['hello! what is your name?',
#            'tell me a joke!',
#            '中国的首都是哪里？',
#            '杭州在哪里？']
prompts = [
    "Consider a sequence of real numbers \( a_1, a_2, a_3, \ldots, a_{2024} \) such that \( a_1 = 1 \) and for \( n \geq 1 \),\n\[ a_{n+1} = \\frac{a_n + 2}{a_n + 1} \]\nDetermine the value of \( a_{2024} \)."
]

for i, prompt in enumerate(prompts):
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sample_params = SamplingParams(
        max_tokens=2048, ignore_eos=False, skip_special_tokens=False
    )
    result = model.generate(text, sampling_params=sample_params)
    response = result[0].outputs[0].text
    print("\n")
    print(f"\nprompt-{i}:", text)
    print(f"response-{i}:", response)
