# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json
import os
import random

import torch
from transformers import AutoTokenizer

from flood.utils.request import Request


class Reader:
    @staticmethod
    def read_sharegpt_dataset(filename, tokenizer_path, max_count=1000):

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # tokenizer = Reader.load_tokenizer(tokenizer_path)

        lines = []
        for line in open(filename, 'r'):
            lines.append(line.strip('\n'))
        lines = json.loads('\n'.join(lines))
        reqs = []
        for i, line in enumerate(lines):
            cs = line['conversations']
            if len(cs) < 2:
                continue
            prompt = cs[0]['value']
            input_length = len(tokenizer(prompt).input_ids)
            if input_length >= 1024 or input_length < 4:
                continue
            response = cs[1]['value']
            output_length = len(tokenizer(response).input_ids)
            if input_length + output_length >= 2048 or output_length < 4:
                continue
            req = Request(i, input_text=prompt, input_length=input_length,
                          output_length=output_length)
            reqs.append(req)
            if len(reqs) >= max_count:
                break
        Reader.sort_by(reqs, key='random')
        Reader.stat(reqs)
        return reqs

    @staticmethod
    def read_fix_dataset(tokenizer_path, prompts=None, max_count=1000, output_length=200):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if prompts is None:
            prompts = ['hello! what is your name?',
                    'tell me a joke!',
                    '中国的首都是哪里？',
                    '杭州在哪里？']
        reqs = []
        for i, prompt in enumerate(prompts[:max_count]):
            chat = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False,
                                                   add_generation_prompt=True)
            input_length = len(tokenizer(prompt).input_ids)
            req = Request(i, input_text=prompt, input_length=input_length,
                          output_length=output_length)
            reqs.append(req)
        Reader.stat(reqs)
        return reqs

    @staticmethod
    def read_dummy_dataset(max_count=1000, input_length=200, output_length=200,
                           flunc=0.0):
        reqs = []
        for i in range(max_count):
            il = input_length + int(
                (2 * random.random() - 1) * flunc * input_length)
            ol = output_length + int(
                (2 * random.random() - 1) * flunc * output_length)
            prompt = 'hi' * il
            req = Request(i, input_text=prompt, input_length=il,
                          output_length=ol)
            reqs.append(req)
        Reader.stat(reqs)
        return reqs

    # @staticmethod
    # def load_tokenizer(tokenizer_path):
    #     filename = os.path.join(tokenizer_path, 'tokenizer_config.json')
    #     conf = json.load(open(filename))
    #     tokenizer_type = conf['tokenizer_class']
    #     if tokenizer_type == 'BailingTokenizer':
    #         from flood.models.tokenization_bailing import BailingTokenizer
    #         tokenizer = BailingTokenizer.from_pretrained(tokenizer_path)
    #     else:
    #         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #     return tokenizer

    @staticmethod
    def get_conf(model_path):
        conf = json.load(open(os.path.join(model_path, 'config.json')))
        model_type = conf['architectures'][0]
        torch_dtype = conf['torch_dtype']
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
            torch_dtype]
        n_layer = conf.get('num_layers') or conf.get('num_hidden_layers')
        hidden_size = conf['hidden_size']
        num_heads = conf['num_attention_heads']
        head_dim = conf.get('head_dim')
        head_dim = head_dim if head_dim else hidden_size // num_heads
        num_key_value_heads = conf['num_key_value_heads']

        return model_type, torch_dtype, n_layer, num_key_value_heads, head_dim

    @staticmethod
    def list_files(data_path, postfix=None):
        file_names = []
        for path, dir_lst, file_lst in os.walk(data_path):
            for file_name in file_lst:
                if postfix is not None and postfix not in file_name:
                    continue
                path_name = os.path.join(path, file_name)
                file_names.append((file_name, path_name))
        file_names = sorted(file_names, key=lambda x: x[0])
        return file_names

    @staticmethod
    def stat(reqs):
        if len(reqs) == 0:
            print('sample:0')
            return
        input_lengths = [x.input_length for x in reqs if x.input_length > 0]
        output_lengths = [x.output_length for x in reqs if x.input_length > 0]
        count = max(len(input_lengths), 1)
        mean_input_length = sum(input_lengths) / count
        mean_output_length = sum(output_lengths) / count
        print(
            f'sample:{len(reqs)} input_length:{min(input_lengths):.0f}/{mean_input_length:.0f}/{max(input_lengths):.0f} ' \
            f'output_length:{min(output_lengths):.0f}/{mean_output_length:.0f}/{max(output_lengths):.0f}')

    @staticmethod
    def sort_by(reqs, key='input', reverse=False):
        if len(reqs) == 0:
            return reqs
        if key == 'input':
            return sorted(reqs, key=lambda x: x.input_length, reverse=reverse)
        elif key == 'random':
            random.shuffle(reqs)
            return reqs
        elif key is None:
            return reqs
        else:
            raise ValueError(f'unknown key:{key}')
