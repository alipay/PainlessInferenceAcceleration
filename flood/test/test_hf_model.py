# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts = [
    "hello! what is your name?",
    "tell me a joke!",
    "中国的首都是哪里？",
    "杭州在哪里？",
][:1]

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
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\nprompt-{i}:", text.replace("\n", "\\n"))
    print(f"response-{i}:", response.replace("\n", "\\n") + "\n")
