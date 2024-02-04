# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import cProfile
import io
import json
import pstats
import random
import sys
import time
from pstats import SortKey

import torch


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Benchmark():
    def __init__(self,
                 log_dir=None,
                 eos=None,
                 eop=None,
                 device='cuda:0'):
        self.log_dir = log_dir
        self.eos = eos
        self.eop = eop  # end token id of prompt, ignore if end token id of prompt is not a fixed id
        self.device = device

        self.model = None
        self.tokenizer = None

        self.prompts = []
        self.answers = []
        self.ids = []

        self.warmup_prompts = []
        self.warmup_answers = []
        self.warmup_ids = []

        self.stop_words = [',', '.', ' ', '\n', '，', ',']
        self.stop_ids = None

        self.logger = None
        if self.log_dir is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.logger = open(f'{self.log_dir}_{timestamp}.log', 'w+')

    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        raise NotImplementedError()

    def save_answers(self, src_dir, dst_dir, max_new_tokens=256, batch_size=1, prompt_name='prompt', response_name='response', max_count=None, use_lookahead=False):
        lines = open(src_dir).readlines()

        prompts = []
        responses = []
        for d in lines:
            d = json.loads(d)
            prompts.append(d[prompt_name])
            responses.append(d[response_name])
            if max_count is not None and len(prompts) >= max_count:
                break

        jsons = []
        qaids = self.generate(prompts, max_new_tokens=max_new_tokens, use_lookahead=use_lookahead,
                              decoding_length=64, branch_length=12,
                              batch_size=batch_size)
        for i, (p, a, ids) in enumerate(qaids):
            r = responses[i]
            jsons.append(json.dumps({'prompt': p, 'response': r, 'pred': a, 'ids': ids}))
        with open(dst_dir, 'w') as f:
            f.write('\n'.join(jsons))

    def load_prompts(self, prompt_dir=None, warmup_prompt_dir=None):
        prompts = []
        answers = []
        for line in open(prompt_dir, 'r'):
            line = json.loads(line)
            prompts.append(line['prompt'])
            answers.append(line.get('answer', None))
        self.prompts = prompts
        self.answers = answers

        if warmup_prompt_dir is not None:
            prompts = []
            answers = []
            ids = []
            for line in open(warmup_prompt_dir, 'r'):
                line = json.loads(line)
                prompts.append(line['prompt'])
                answers.append(line.get('answer', None))
                ids.append(line.get('ids', None))
            self.warmup_prompts = prompts
            self.warmup_answers = answers
            self.warmup_ids = ids

    def tokenize(self, prompt, max_length=256):
        if isinstance(prompt, list):
            inputs = self.tokenizer(prompt,
                                    padding=True,
                                    truncation=False,
                                    return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = None
        return input_ids, position_ids, attention_mask

    def chat(self, prompt, max_new_tokens=256, use_lookahead=False, decoding_length=64, branch_length=8,
             decoding_mode='hier', debug_lookahead=False, max_query_length=2):
        if use_lookahead and decoding_length > 1 and branch_length > 0:
            max_gen_length = max_new_tokens + decoding_length + 1
        else:
            max_gen_length = max_new_tokens
        input_ids, position_ids, attention_mask = self.tokenize(prompt, max_length=max_gen_length)
        tokenizer = self.tokenizer
        model = self.model

        decoding_kwargs = {"use_lookahead": use_lookahead,
                        "debug_lookahead": debug_lookahead,
                        "decoding_mode": decoding_mode,
                        "decoding_length": decoding_length,
                        "branch_length": branch_length,
                        "max_query_length": max_query_length,
                        "stop_words": self.stop_ids}
        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 pad_token_id=self.eos,
                                 eos_token_id=self.eos,
                                 use_cache=True,
                                 max_new_tokens=max_new_tokens,
                                 repetition_penalty=1.1,
                                 do_sample=False,
                                 decoding_kwargs=decoding_kwargs,
                                 return_dict_in_generate=True
                                 )
        output_ids = outputs.sequences
        kwargs = outputs.kwargs if hasattr(outputs, 'kwargs') else {}
        input_length = input_ids.size(-1)
        output_ids = output_ids[:, input_length:].tolist()
        # output_ids = output_ids.tolist()
        output_texts = []
        output_id_list = []
        for token_ids in output_ids:
            output_id_list.append(token_ids)
            text = tokenizer.decode(token_ids)
            output_texts.append(text)
        input_id_list = input_ids.tolist()
        # input_texts = tokenizer.batch_decode(input_ids)
        return prompt, input_id_list, output_id_list, output_texts, kwargs

    def warm_up(self, ids, branch_length=8, eop=None):
        ts = time.time()
        lookahead_cache = self.model.lookahead_cache

        for i, ids_ in enumerate(ids):
            if ids_ is None:
                continue
            lookahead_cache.put([eop] + ids_ if eop else ids_, branch_length=branch_length + 1, mode='output',
                                idx=-1)
            if (i + 1) % 1000 == 0:
                print(f'warmup:{i + 1}, elapse:{round(time.time() - ts, 1)}s')

    def generate(self, qs, use_lookahead=True, max_new_tokens=256, decoding_length=64, branch_length=8, batch_size=16):
        chat_count = len(qs)
        qas = []
        ts = time.time()
        for i in range((chat_count - 1) // batch_size + 1):
            queries = qs[i * batch_size:(i + 1) * batch_size]
            input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(queries,
                                                                                         max_new_tokens=max_new_tokens,
                                                                                         use_lookahead=use_lookahead,
                                                                                         decoding_length=decoding_length,
                                                                                         branch_length=branch_length)
            for j in range(len(queries)):
                qas.append((queries[j], output_texts[j], output_id_list[j]))
            if (i + 1) % 10 == 0:
                print(f'generate:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        return qas

    def batch_chat(self, qs, max_new_tokens=256, decoding_length=64, branch_length=8, decoding_mode='hier',
                   debug_lookahead=False, erase=True, batch_size=1, max_query_length=2):
        total_out_tokens = [0, 0]
        total_times = [0, 0]
        lookahead_cache = self.model.lookahead_cache
        if erase:
            lookahead_cache.fresh()
        chat_count = len(qs)
        for i in range(chat_count // batch_size):
            query = qs[i * batch_size:(i + 1) * batch_size]
            speeds = []
            for j, use_lookahead in enumerate([False, True]):
                in_char = 0
                in_token = 0
                out_char = 0
                out_token = 0
                ts = time.time()
                input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(query,
                                                                                             max_new_tokens=max_new_tokens,
                                                                                             use_lookahead=use_lookahead,
                                                                                             decoding_length=decoding_length,
                                                                                             branch_length=branch_length,
                                                                                             decoding_mode=decoding_mode,
                                                                                             debug_lookahead=debug_lookahead,
                                                                                             max_query_length=max_query_length)
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
                prefix = 'lookahead:' + ('On ' if use_lookahead else 'Off')
                speedup = speeds[-1] / speeds[0] if use_lookahead else 0.0
                print(
                    f"{prefix} mode:{decoding_mode} idx:{i} "
                    f"input:{in_char:.1f}/{in_token:.1f} output:{out_char:.1f}/{out_token:.1f} "
                    f"edl:{edl:.3f}/{dl:.3f}/{et:.3f} time:{t:.3f} speed:{speed_token:.1f} speedup:{speedup:.3f}\n")
        org_speed = total_out_tokens[0] / total_times[0]
        opt_speed = total_out_tokens[1] / total_times[1]
        speedup = opt_speed / org_speed
        print(f'speed:{org_speed:.2f}->{opt_speed:.2f} speedup:{speedup:.3f}')

    def perf_check(self, queries, warmup_ids=None, max_new_tokens=256, sizes=(31, 64),
                   lens=(4, 8, 12), decoding_mode='hier',
                   batch_size=1, max_node_rate=32,max_query_length=2):
        wc = len(warmup_ids) if warmup_ids is not None else 0
        log_str = f'\nmode:{decoding_mode} bs:{batch_size} queries:{len(queries)} warmup:{wc} sizes:{sizes} lens:{lens}'
        print(log_str)
        if batch_size > 1:
            queries = sorted(queries, key=lambda x: len(x))
        speeds = []
        outputs = {}
        lookahead_cache = self.model.lookahead_cache
        for i, decoding_length in enumerate(sizes):
            for j, branch_length in enumerate(lens):
                if decoding_length < branch_length * batch_size:
                    continue
                use_lookahead = decoding_length > 1 and branch_length > 0
                in_char = 0
                in_token = 0
                out_char = 0
                out_token = 0
                dls = []
                edls = []
                fts = []
                if use_lookahead:
                    lookahead_cache.fresh()
                    lookahead_cache.max_node = max_node_rate * decoding_length
                    lookahead_cache.max_output_node = max_node_rate * decoding_length // 2
                    if warmup_ids is not None:
                        self.warm_up(warmup_ids, branch_length=branch_length, eop=self.eop)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=None)
                ts = time.time()
                n_b = len(queries) // batch_size
                times = []
                for k in range(n_b):
                    qs_ = queries[k * batch_size:(k + 1) * batch_size]
                    ts_ = time.time()
                    input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(qs_,
                                                                                                 max_new_tokens=max_new_tokens,
                                                                                                 use_lookahead=use_lookahead,
                                                                                                 decoding_length=decoding_length,
                                                                                                 branch_length=branch_length,
                                                                                                 decoding_mode=decoding_mode,
                                                                                                 max_query_length=max_query_length)
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
                        log_str = f'mode:{decoding_mode} step:{k + 1} ' \
                                  f'decoding:{decoding_length}/{branch_length} bs:{batch_size} ' \
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
                outputs[(decoding_length, branch_length)] = speed
                # print(f"Human:{query}")
                # print(f"Robot:{results[0]}")
                dl = sum(dls) / max(len(dls), 1)
                edl = sum(edls) / max(len(edls), 1)
                ft = sum(fts) / max(len(fts), 1)
                ms = torch.cuda.memory_stats()
                mem = ms['reserved_bytes.large_pool.peak'] / 1024 ** 3
                speedup = speeds[-1] / speeds[0]
                times = [round(x, 3) for x in times]
                log_str = f"mode:{decoding_mode} bs:{batch_size} " \
                          f"decoding_length:{decoding_length} branch_length:{branch_length} " \
                          f"query:{len(queries)} warmup:{wc} " \
                          f"input:{in_token:.1f} output:{out_token:.1f} " \
                          f"edl:{edl:.3f}/{dl:.3f}/{ft:.3f} time:{t:.3f} " \
                          f"speed:{speed:.1f} mem:{mem:.3f} "
                print(log_str)
                if self.logger is not None:
                    self.logger.write(log_str + '\n')
                    self.logger.flush()

        return outputs

    def naive_profile(self, qs, use_lookahead=False, count=64, sortby=0):
        pr = cProfile.Profile()
        pr.enable()
        for q in qs:
            self.chat(q, use_lookahead=use_lookahead)
        pr.disable()
        s = io.StringIO()
        if sortby == 0:
            sortby = SortKey.TIME
        else:
            sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(count)
        print(s.getvalue())

    def naive_prof_trie(self, warmup_ids, decoding_length=64, branch_length=24, edl=8, put_count=10000, get_count=100,
                        count=64, sortby=SortKey.TIME, put=True, get=True):
        pr = cProfile.Profile()
        pr.enable()
        lookahead_cache = self.model.lookahead_cache
        if put:
            lookahead_cache.fresh()
            ts = time.time()
            for i, ids_ in enumerate(warmup_ids[:put_count]):
                lookahead_cache.put(ids_, branch_length=branch_length + 1, mode='output', idx=-1, final=True)
                if (i + 1) % 1000 == 0:
                    print(f'prof put:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        if get:
            ts = time.time()
            for i, ids_ in enumerate(warmup_ids[:get_count]):
                for j in range(0, len(ids_) - 1, edl):
                    lookahead_cache.bat_get([ids_[j:j + 2]], decoding_length=decoding_length,
                                            branch_length=branch_length, decoding_cursors=[j], mode='mix',
                                            indices=[0], decoding_mode='hier')
                if (i + 1) % 1000 == 0:
                    print(f'prof get:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        pr.disable()
        s = io.StringIO()
        # sortby = SortKey.CUMULATIVE SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(count)
        print(s.getvalue())

    def torch_profile(self, use_lookahead=False):

        # pip install torch_tb_profiler
        # tensorboard --logdir=./prof
        # http://localhost:6006/#pytorch_profiler

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof'),
            record_shapes=True,
            with_stack=True)
        prof.start()
        for p in self.prompts:
            prof.step()
            self.chat(p, use_lookahead=use_lookahead)
        prof.stop()

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

    def grid_search(self, chat_count=100, warmup_count=10000):

        ps = self.prompts
        warmup_ids = self.warmup_ids
        outputs = self.perf_check(ps[:chat_count],
                                  warmup_ids=warmup_ids[:warmup_count],
                                  sizes=[16 * x - 1 for x in [1, 2, 4, 8, 16]],
                                  lens=[4 * x for x in range(1, 11)],
                                  batch_size=1)

        opt_size, opt_len = sorted(outputs.items(), key=lambda x: x[1], reverse=True)[0][0]
        for rate in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            self.perf_check(ps[:chat_count], warmup_ids=ids[:warmup_count], sizes=[opt_size],
                            lens=[opt_len], max_node_rate=rate)

    def batch_grid_search(self, chat_count=100, warmup_count=10000):

        ps = self.prompts
        warmup_ids = self.warmup_ids
        decoding_mode = 'hier'
        for bs in [2, 4, 6, 8]:
            outputs = self.perf_check(ps[:chat_count],
                                      warmup_ids=warmup_ids[:warmup_count],
                                      sizes=[16 * x - bs for x in [8, 16]],
                                      lens=[4, 8, 12, 16],
                                      batch_size=bs,
                                      decoding_mode=decoding_mode)

            opt_size, opt_len = sorted(outputs.items(), key=lambda x: x[1], reverse=True)[0][0]
            self.perf_check(ps[:chat_count], warmup_ids=warmup_ids[:warmup_count], sizes=[opt_size],
                            lens=[opt_len], batch_size=bs)
