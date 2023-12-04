# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import time
from operator import itemgetter
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import copy
import types
import json
import random
import warnings
import pandas as pd
import os
import sys
import cProfile, pstats, io
from pstats import SortKey
import torch

sys.path.append('../../lookahead')
from common.pretrained_model import PrefetchCache

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class Benchmark():
    def __init__(self,
                 model_type='opt',
                 model_name='opt_1b3',
                 model_dir=None,
                 prompt_dir=None,
                 ):
        self.model_type = model_type
        self.model_name = model_name
        self.model_dir = model_dir
        self.prompt_dir = prompt_dir

        self.model = None
        self.tokenizer = None
        self.eos = None
        self.prefetch_cache = None

        self.prompts = []
        self.answers = []
        self.ids = []

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.logger = open(f'perf_{model_type}_{model_name}_{timestamp}.log', 'w+')

    def load_model(self):
        ts = time.time()
        model_type = self.model_type
        method_name = '_load_' + model_type
        getattr(self, method_name)()

        self.model.cuda().eval()
        print(f'load model in {round(time.time() - ts, 3)}s')

        self.model.prefetch_cache = self.prefetch_cache


    def _load_glm(self):
        from models.antglm.tokenization_glm_ext import GLMChineseTokenizer
        from models.antglm.modeling_glm_batch import GLMForConditionalGeneration
        model_dir = self.model_dir
        self.tokenizer = GLMChineseTokenizer.from_pretrained(model_dir)
        self.model = GLMForConditionalGeneration.from_pretrained(model_dir
                                                                 , cache_dir='./'
                                                                 , torch_dtype=torch.float16
                                                                 , low_cpu_mem_usage=True
                                                                 , device_map='auto'
                                                                 )
        self.prefetch_cache = PrefetchCache(eos=50005, stop_words={43359, 43360, 43361, 43362})

    def _load_chatglm(self):
        from models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
        from models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
        assert not ChatGLMForConditionalGeneration._batch_prefetch
        model_dir = self.model_dir
        self.tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
        self.model = ChatGLMForConditionalGeneration.from_pretrained(model_dir
                                                                     , cache_dir='./'
                                                                     , torch_dtype=torch.float16
                                                                     , low_cpu_mem_usage=True
                                                                     , device_map='auto'
                                                                     )
        self.prefetch_cache = PrefetchCache(eos=tokenizer.eos_token_id, stop_words={1919, 869, 259, 1577})

    def _load_llama(self):
        from models.modeling_llama import LlamaForCausalLM
        assert not LlamaForCausalLM._batch_prefetch
        model_dir = self.model_dir
        self.model = LlamaForCausalLM.from_pretrained(model_dir
                                                      , cache_dir='./'
                                                      , torch_dtype=torch.float16
                                                      , low_cpu_mem_usage=True
                                                      , device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.prefetch_cache = PrefetchCache(eos=self.tokenizer.eos_token_id, stop_words={1919, 869, 259, 1577})

    def _load_bloom(self):
        from models.modeling_bloom import BloomForCausalLM
        assert not BloomForCausalLM._batch_prefetch
        model_dir = self.model_dir
        self.model = BloomForCausalLM.from_pretrained(model_dir
                                                      , cache_dir='./'
                                                      , torch_dtype=torch.float16
                                                      , low_cpu_mem_usage=True
                                                      , device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefetch_cache = PrefetchCache(eos=self.tokenizer.eos_token_id)

    def _load_baichuan(self):

        from models.baichuan2.modeling_baichuan import BaichuanForCausalLM
        assert not BaichuanForCausalLM._batch_prefetch
        model_dir = self.model_dir
        self.model = BaichuanForCausalLM.from_pretrained(model_dir
                                                         , cache_dir='./'
                                                         , torch_dtype=torch.float16
                                                         , low_cpu_mem_usage=True
                                                         , device_map='auto'
                                                         )
        from models.baichuan2.tokenization_baichuan import BaichuanTokenizer
        self.tokenizer = BaichuanTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefetch_cache = PrefetchCache(eos=self.tokenizer.eos_token_id)

    def _load_gptj(self):
        from models.modeling_gptj import GPTJForCausalLM
        assert not GPTJForCausalLM._batch_prefetch
        model_dir = self.model_dir
        self.model = GPTJForCausalLM.from_pretrained(model_dir
                                                , cache_dir='./'
                                                , torch_dtype=torch.float16
                                                , low_cpu_mem_usage=True
                                                , device_map='auto'
                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefetch_cache = PrefetchCache(eos=self.tokenizer.eos_token_id)

    def _load_qwen(self):
        from models.qwen.modeling_qwen import QWenLMHeadModel
        from models.qwen.tokenization_qwen import QWenTokenizer
        model_dir = self.model_dir
        self.tokenizer = QWenTokenizer.from_pretrained(model_dir)
        self.model = QWenLMHeadModel.from_pretrained(model_dir
                                                , cache_dir='./'
                                                , torch_dtype=torch.float16
                                                , low_cpu_mem_usage=True
                                                , device_map='auto'
                                                )
        self.tokenizer.pad_token = '<|endoftext|>'
        self.tokenizer.eos_token = '<|endoftext|>'
        self.tokenizer.padding_side = 'left'
        self.prefetch_cache = PrefetchCache(eos=self.tokenizer.eos_token_id)

    def _load_opt(self):
        from models.modeling_opt import OPTForCausalLM
        assert not OPTForCausalLM._batch_prefetch
        model_dir = self.model_dir
        self.model = OPTForCausalLM.from_pretrained(model_dir
                                                    , cache_dir='./'
                                                    , torch_dtype=torch.float16
                                                    , low_cpu_mem_usage=True
                                                    , device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.prefetch_cache = PrefetchCache(eos=self.tokenizer.eos_token_id)

    def save_answers(self, src_dir, dst_dir, max_length=256, batch_size=1, max_count=None, use_predict=False):
        lines = open(src_dir).readlines()

        prompts = []
        for d in lines:
            d = json.loads(d)
            prompts.append(d['prompt'])
            if max_count is not None and len(prompts) >= max_count:
                break

        jsons = []
        qaids = self.generate(prompts, max_length=max_length, use_prefetch=use_predict,
                              prefetch_size=63, prefetch_length=12,
                              batch_size=batch_size)
        for p, a, ids in qaids:
            jsons.append(json.dumps({'prompt': p, 'answer': a, 'ids': ids}))
        with open(dst_dir, 'w') as f:
            f.write('\n'.join(jsons))

    def load_prompts(self):
        prompts = []
        answers = []
        ids = []
        for line in open(self.prompt_dir, 'r'):
            line = json.loads(line)
            prompts.append(line['prompt'])
            answers.append(line.get('answer', None))
            ids.append(line.get('ids', None))
        self.prompts = prompts
        self.answers = answers
        self.ids = ids

    def shuffle_prompts(self, ps, ans, ids, min_length=20):
        indices = list(range(len(ps)))
        random.Random(4).shuffle(indices)
        indices = [i for i, x in enumerate(ids) if len(x) >= min_length]
        ps = [ps[i] for i in indices]
        ans = [ans[i] for i in indices]
        ids = [ids[i] for i in indices]
        return ps, ans, ids

    def chat(self, prompt, max_length=256, use_prefetch=False, prefetch_size=63, prefetch_length=8,
             prefetch_mode='trie', debug_prefetch=False):
        model_type = self.model_type
        tokenizer = self.tokenizer
        model = self.model
        if model_type == 'glm':
            if isinstance(prompt, list):
                prompt = [x if '[gMASK]' in x else x + '[gMASK]' for x in prompt]
                inputs = tokenizer(prompt,
                                   padding=True,
                                   truncation=False,
                                   return_tensors="pt")
            else:
                if '[gMASK]' not in prompt:
                    prompt = prompt + '[gMASK]'
                inputs = tokenizer(prompt,
                                   padding=True,
                                   truncation=False,
                                   return_tensors="pt",
                                   )

            inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=max_length + prefetch_size + 3)
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['generation_attention_mask'].cuda()
            position_ids = inputs['position_ids'].cuda()
            self.eos = 50005
        elif model_type == 'chatglm':
            inputs = model.build_inputs(tokenizer, prompt, history=[])
            input_ids = inputs.input_ids.cuda()
            attention_mask = inputs.attention_mask.cuda()
            position_ids = None
            self.eos = 2
        else:
            if isinstance(prompt, list):
                inputs = tokenizer(prompt,
                                   padding=True,
                                   truncation=False,
                                   return_tensors="pt")
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.cuda()
            attention_mask = inputs.attention_mask.cuda()
            position_ids = None
            self.eos = tokenizer.eos_token_id

        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=self.eos,
                                 use_cache=True,
                                 max_new_tokens=max_length,
                                 repetition_penalty=1.0,
                                 do_sample=False,
                                 use_prefetch=use_prefetch,
                                 prefetch_size=prefetch_size,
                                 prefetch_length=prefetch_length,
                                 debug_prefetch=debug_prefetch,
                                 return_dict_in_generate=True
                                 )
        output_ids = outputs.sequences
        kwargs = outputs.scores if outputs.scores is not None else {}
        input_length = input_ids.size(-1)
        output_ids = output_ids[:, input_length:].tolist()
        output_texts = []
        output_id_list = []
        for token_ids in output_ids:
            output_id_list.append(token_ids)
            text = tokenizer.decode(token_ids)
            output_texts.append(text)
        input_id_list = input_ids.tolist()
        input_texts = tokenizer.batch_decode(input_ids)
        return input_texts, input_id_list, output_id_list, output_texts, kwargs

    def warm_up(self, ids, prefetch_length=8, use_sop=True, sop=50006):
        ts = time.time()
        prefetch_cache = self.prefetch_cache

        for i, ids_ in enumerate(ids):
            if ids_ is None:
                continue
            prefetch_cache.put([sop] + ids_ if use_sop else ids_, prefetch_length=prefetch_length + 1, mode='output',
                               idx=-1)
            if (i + 1) % 1000 == 0:
                print(f'warmup:{i + 1}, elapse:{round(time.time() - ts, 1)}s')

    def generate(self, qs, use_prefetch=True, max_length=256, prefetch_size=63, prefetch_length=8, batch_size=16):
        chat_count = len(qs)
        qas = []
        ts = time.time()
        for i in range((chat_count - 1) // batch_size + 1):
            queries = qs[i * batch_size:(i + 1) * batch_size]
            input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(queries,
                                                                                    max_length=max_length,
                                                                                    use_prefetch=use_prefetch,
                                                                                    prefetch_size=prefetch_size,
                                                                                    prefetch_length=prefetch_length)
            for j in range(len(queries)):
                qas.append((queries[j], output_texts[j], output_id_list[j]))
            if (i + 1) % 10 == 0:
                print(f'generate:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        return qas

    def batch_chat(self, qs, max_length=256, prefetch_size=63, prefetch_length=8, 
                   debug_prefetch=False, erase=True, batch_size=1):
        total_out_tokens = [0, 0]
        total_times = [0, 0]
        prefetch_cache = self.prefetch_cache
        if erase:
            prefetch_cache.fresh()
        chat_count = len(qs)
        for i in range(chat_count // batch_size):
            query = qs[i * batch_size:(i + 1) * batch_size]
            speeds = []
            for j, use_prefetch in enumerate([False, True]):
                in_char = 0
                in_token = 0
                out_char = 0
                out_token = 0
                ts = time.time()
                input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(query,
                                                                                        max_length=max_length,
                                                                                        use_prefetch=use_prefetch,
                                                                                        prefetch_size=prefetch_size,
                                                                                        prefetch_length=prefetch_length,
                                                                                        debug_prefetch=debug_prefetch)
                in_char += sum([len(x) for x in input_texts])
                in_token += sum([len(x) for x in input_id_list])
                out_char += sum([len(x) for x in output_texts])
                out_token += sum([len(x) for x in output_id_list])
                t = (time.time() - ts)
                speed_char = out_char / t
                speed_token = out_token / t
                speeds.append(speed_token)
                total_out_tokens[j] += out_token
                total_times[j] += t
                bs = len(query)
                dls = kwargs.get('dls', [])
                dl = sum(dls[bs:]) / len(dls[bs:]) if len(dls) > bs else 0.0
                edls = kwargs.get('edls', [])
                edl = sum(edls[bs:]) / len(edls[bs:]) if len(edls) > bs else 0.0
                et = kwargs.get('fts', [0])[0]
                # print(f"Human:{query[:80]}...")
                print(f"1/{bs} Robot:{output_texts[0]}")
                prefix = 'Prefetch:' + ('On ' if use_prefetch else 'Off')
                imp = speeds[-1] / speeds[0] if use_prefetch else 0.0
                print(
                    f"{prefix} model:{self.model_type} idx:{i}/{chat_count} "
                    f"input:{in_char:.1f}/{in_token:.1f} output:{out_char:.1f}/{out_token:.1f} "
                    f"edl:{edl:.3f}/{dl:.3f}/{et:.3f} time:{t:.3f} speed:{speed_token:.1f} imp:{imp:.3f}\n")
        org_speed = total_out_tokens[0] / total_times[0]
        opt_speed = total_out_tokens[1] / total_times[1]
        imp = opt_speed / org_speed
        print(f'speed:{org_speed:.2f}->{opt_speed:.2f} imp:{imp:.3f}')

    def perf_check(self, queries, warmup_ids=None, max_length=256, sizes=(31, 63),
                   lens=(4,8,12), 
                   batch_size=1, max_node_rate=32):
        wc = len(warmup_ids) if warmup_ids is not None else 0
        log_str = f'\nmodel:{model_type} bs:{batch_size} queries:{len(queries)} warmup:{wc} sizes:{sizes} lens:{lens}'
        print(log_str)
        if batch_size > 1:
            queries = sorted(queries, key=lambda x: len(x))
        speeds = []
        outputs = {}
        visited = False
        prefetch_cache = self.prefetch_cache
        for i, prefetch_size in enumerate(sizes):
            for j, prefetch_length in enumerate(lens):
                if prefetch_size < prefetch_length * batch_size:
                    continue
                if prefetch_size == 0 or prefetch_length == 0:
                    if visited:
                        continue
                    visited = True
                use_prefetch = prefetch_size > 0 and prefetch_length > 0
                in_char = 0
                in_token = 0
                out_char = 0
                out_token = 0
                dls = []
                edls = []
                fts = []
                sop = 50006 if model_type == 'glm' else 2
                if use_prefetch:
                    prefetch_cache.fresh()
                    prefetch_cache.max_node = max_node_rate * prefetch_size
                    prefetch_cache.max_output_node = max_node_rate * prefetch_size // 2
                    if warmup_ids is not None:
                        self.warm_up(warmup_ids, prefetch_length=prefetch_length, use_sop=True, sop=sop)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=None)
                ts = time.time()
                n_b = len(queries) // batch_size
                times = []
                for k in range(n_b):
                    qs_ = queries[k * batch_size:(k + 1) * batch_size]
                    ts_ = time.time()
                    input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(qs_, max_length=max_length,
                                                                                            use_prefetch=use_prefetch,
                                                                                            prefetch_size=prefetch_size,
                                                                                            prefetch_length=prefetch_length)
                    te_ = time.time()
                    times.append(te_ - ts_)
                    in_char += sum([len(x) for x in qs_])
                    in_token += sum([len(x) for x in input_id_list])
                    out_char += sum([len(x) for x in output_texts])
                    out_token += sum([len(x) for x in output_id_list])
                    bs = len(qs_)
                    dls_ = kwargs.get('dls', [])
                    dls.extend(dls_[bs:] if len(dls_) > bs else [])
                    edls_ = kwargs.get('edls', [])
                    edls.extend(edls_[bs:] if len(edls_) > bs else [])
                    fts.append(kwargs.get('fts', [0])[0])
                    if (k + 1) % (100 // batch_size) == 0:
                        elapse = time.time() - ts
                        speed = out_token / elapse
                        avg_in_token = float(in_token) / (k + 1) / batch_size
                        avg_out_token = float(out_token) / (k + 1) / batch_size
                        dl = sum(dls) / max(len(dls), 1)
                        edl = sum(edls) / max(len(edls), 1)
                        ft = sum(fts) / max(len(fts), 1)
                        log_str = f'model:{self.model_type} step:{k + 1} ' \
                                  f'prefetch:{prefetch_size}/{prefetch_length} bs:{batch_size} ' \
                                  f'elapse:{elapse:.1f}s in:{avg_in_token:.1f} out:{avg_out_token:.1f} ' \
                                  f'edl:{edl:.3f}/{dl:.3f}/{ft:.3f} speed:{speed:.1f}token/s'
                        print(log_str)
                n_repeat = len(queries)
                in_char /= n_repeat
                in_token /= n_repeat
                out_char /= n_repeat
                out_token /= n_repeat
                t = (time.time() - ts) / n_repeat
                speed = out_token / t
                speeds.append(speed)
                outputs[(prefetch_size, prefetch_length)] = speed
                # print(f"Human:{query}")
                # print(f"Robot:{results[0]}")
                dl = sum(dls) / max(len(dls), 1)
                edl = sum(edls) / max(len(edls), 1)
                ft = sum(fts) / max(len(fts), 1)
                ms = torch.cuda.memory_stats()
                mem = ms['reserved_bytes.large_pool.peak'] / 1024 ** 3
                imp = speeds[-1] / speeds[0]
                log_str = f"model:{model_type} bs:{batch_size} " \
                          f"prefetch:{prefetch_size}/{prefetch_length} " \
                          f"query:{len(queries)} warmup:{wc} " \
                          f"input:{in_token:.1f} output:{out_token:.1f} " \
                          f"edl:{edl:.3f}/{dl:.3f}/{ft:.3f} time:{t:.3f} " \
                          f"speed:{speed:.1f} imp:{imp:.3f} mem:{mem:.3f} " \
                          f"max_node:{max_node_rate}"
                print(log_str)
                if self.logger is not None:
                    self.logger.write(log_str + '\n')
                    self.logger.flush()

        return outputs

    def naive_profile(self, qs, use_prefetch=False, count=64, sortby=SortKey.TIME):
        pr = cProfile.Profile()
        pr.enable()
        for q in qs:
            self.chat(q, use_prefetch=use_prefetch)
        pr.disable()
        s = io.StringIO()
        # sortby = SortKey.CUMULATIVE SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(count)
        print(s.getvalue())

    def naive_prof_trie(self, warmup_ids, prefetch_size=63, prefetch_length=24, edl=8, put_count=10000, get_count=100,
                        count=64, sortby=SortKey.TIME, put=True, get=True):
        pr = cProfile.Profile()
        pr.enable()
        prefetch_cache = self.prefetch_cache
        if put:
            prefetch_cache.fresh()
            ts = time.time()
            for i, ids_ in enumerate(warmup_ids[:put_count]):
                prefetch_cache.put(ids_, prefetch_length=prefetch_length + 1, mode='output', idx=-1, final=True)
                if (i + 1) % 1000 == 0:
                    print(f'prof put:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        if get:
            ts = time.time()
            for i, ids_ in enumerate(warmup_ids[:get_count]):
                for j in range(0, len(ids_) - 1, edl):
                    prefetch_cache.bat_get([ids_[j:j + 2]], prefetch_size=prefetch_size,
                                           prefetch_length=prefetch_length, prefetch_cursors=[j], mode='mix',
                                           indices=[0])
                if (i + 1) % 1000 == 0:
                    print(f'prof get:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        pr.disable()
        s = io.StringIO()
        # sortby = SortKey.CUMULATIVE SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(count)
        print(s.getvalue())

    def torch_profile(self,use_prefetch=False):
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof'),
            record_shapes=True,
            with_stack=True)
        prof.start()
        for p in self.prompts:
            prof.step()
            self.chat(p, use_prefetch=use_prefetch)
        prof.stop()
        # pip install torch_tb_profiler
        # tensorboard --logdir=./prof
        # http://localhost:6006/#pytorch_profiler

    def to_words(self, token_ids):
        if isinstance(token_ids, list):
            tokens = []
            for i in token_ids:
                tokens.append(self.tokenizer._convert_id_to_token(i))
            print(tokens)
        else:
            print(self.tokenizer._convert_id_to_token(token_ids))

    def to_ids(self, tokens):
        return self.tokenizer._convert_token_to_id(tokens)

    def grid_search(self,n_repeat=1):
        
        ids = self.ids
        ps = self.prompts
        # 测试1：参数网格搜索
        outputs = self.perf_check(ps[:chat_count], 
                                       warmup_ids=ids[chat_count:chat_count+warmup_count], 
                                       sizes=[16*x-1 for x in [1,2,4,8,16]], 
                                       lens=[4*x for x in range(1,11)],  
                                       batch_size=1)
        
        # 测试2: 最佳参数执行
        opt_size, opt_len = sorted(outputs.items(), key=lambda x:x[1], reverse=True)[0][0]
        for _ in range(n_repeat):
            self.perf_check(ps[:chat_count], warmup_ids=ids[chat_count:chat_count+warmup_count], sizes=[opt_size], lens=[opt_len])
        

model_type = 'llama'
model_dir = '/mntnlp/common_base_model/llama2-13b-chat'
prompt_dir = '/ossfs/workspace/pia/lookahead/tests/datasets/dolly_10k_llama2_13b_chat.jsonl'
worker = Benchmark(model_type=model_type,
                   model_dir=model_dir,
                   prompt_dir=prompt_dir,
                   )
worker.load_model()
# worker.save_answers(src_dir='', dst_dir=prompt_dir, max_length=256, batch_size=1,max_count=10)
worker.load_prompts()

chat_count = 1000
warmup_count = 15000

""" serving mode """
# case study
prompt = "Hello, I'm am conscious and"
worker.chat(prompt, 
            max_length=128, 
            use_prefetch=False, 
            prefetch_size=15, 
            prefetch_length=4, 
            debug_prefetch=False)

# check the different between greedy generation and lookahead generation
worker.batch_chat(worker.prompts[:10], 
                  max_length=128, 
                  prefetch_size=15, 
                  prefetch_length=4, 
                  debug_prefetch=False,
                  erase=True, 
                  batch_size=1)

# performance evaluation with lookahead and without lookahead
worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.ids[chat_count:chat_count + warmup_count], sizes=[63], lens=[0,12],
           max_length=128)


