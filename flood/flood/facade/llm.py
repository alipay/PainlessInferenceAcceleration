# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import asyncio
import copy
import gc
import json
import math
import os
import random
import time
from collections import OrderedDict
from multiprocessing.sharedctypes import Array
from typing import List, Optional, Tuple, Union
import itertools

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from transformers import AutoTokenizer


from flood.layers.linear import NativeLinear
from flood.layers.sync import TaskSyncLayer
from flood.utils.speculative import Lookahead
from flood.utils.batch import Batch, Slot
from flood.utils.cache import SegmentCache
from flood.utils.reader import Reader
from flood.utils.request import Req, Request
from flood.models import model_class_map, model_attr_map

random.seed(7)
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


os.environ['TOKENIZERS_PARALLELISM'] = 'false'



class OutputQueue:
    def __init__(self, queue, index=0, idle=True):
        self.queue = queue
        self.index = index
        self.idle = idle


def log(logger, line):
    print(line, flush=True)
    if logger is not None:
        logger.write(line + '\n')
        logger.flush()


class LLM():
    def __init__(self,
                 model_path: str,
                 model_dtype: Optional[torch.dtype] = None,
                 head_dtype: Optional[torch.dtype] = None,
                 emb_dtype: Optional[torch.dtype] = None,
                 cache_dtype: Optional[torch.dtype] = None,
                 n_stage: Optional[int] = None,
                 n_proc: Optional[int] = None,
                 cache_size: Optional[Union[int, float]] = None,
                 slot_size: int = 8192,
                 schedule_mode: str = 'pingpong',
                 chunk_size: int = 1024,
                 sync_wait_time: Tuple = (4, 4),
                 queue_timeout: float = 0.001,
                 max_slot_alloc_fail_count: int = 1,
                 alloc_early_exit_rate: float = 0.95,
                 slot_first_alloc_rate: float = 1.0,
                 slot_fully_alloc_under: Optional[int] = None,
                 batch_size_step: Optional[int] = None,
                 min_batch_size: int = 16,
                 max_batch_size: int = 512,
                 batch_size_round_frac: float = 0.0,
                 min_decode_rate: float = 1.0,
                 eos_token_id: Optional[Tuple] = None,
                 embedding_dir: Optional[str] = None,
                 spec_algo: Optional[str] = None,
                 kernels: Tuple = ('sa',),
                 output_file_name: str = 'tmp.jsonl',
                 output_file_mode: str = 'w+',
                 output_field_names: Optional[Tuple] = None,
                 logger: str = 'tmp.log',
                 debug: bool = False):
        """
        main class for inference
        :param model_path: HuggingFace-style model path
        :param model_dtype: dtype for linear layers.
                None: inherit from model_dtype in config.json
                torch.float8_e4m3fn: autocast linear layer dtype
                    to torch.float8_e4m3fn with channel-wise quantization.
        :param head_dtype: dtype for lm_head layer.
                None: inherit from model_dtype in config.json
                torch.float8_e4m3fn: autocast linear layer dtype
                    to torch.float8_e4m3fn with channel-wise quantization.
        :param emb_dtype: dtype for token_embeds layer.
                None: inherit from model_dtype in config.json
                torch.float8_e4m3fn: autocast linear layer dtype
                    to torch.float8_e4m3fn with channel-wise quantization.
        :param cache_dtype: dtype for kvcache.
                None: inherit from model_dtype in config.json
                torch.float8_e4m3fn: use fp8 attention (WIP).
        :param n_stage: pipeline stages. Can not larger than number of GPU.
                None: number of GPU.
        :param n_proc: number of process.
                None: n_stage+1
        :param cache_size: kv_cache capacity.
                Int: number of tokens.
                Float: max percent of GPU memory usage.
                None: determined heuristically.
        :param slot_size: total segment count.
        :param schedule_mode: schedule mode.
                pingpong: do prefill until no slot available,
                        and do decoding until batch size decrease.
                mix: mix of prefill and decoding (Not fully tested).
                timely: do prefill as soon as possible (Not fully tested).
        :param chunk_size: chunk size for prefill.
        :param sync_wait_time: sync time (second).
        :param queue_timeout: queue timeout (second).
        :param max_slot_alloc_fail_count: max slot alloc fail count, if reach
                    the count, it will switch from prefill to decode.
        :param alloc_early_exit_rate: rate of allocation size and segment size.
                    will return a segment without traversing all the segments
                     if the rate exceed this value.
        :param slot_first_alloc_rate: the segment size for output tokens
                    in the first allocation.
        :param slot_fully_alloc_under: output length under this value will be
                    fully allocated.
        :param batch_size_step: batch size rounding for decoding.
                    None: round to power of 2
                    Int: round to multiple of this value.
        :param min_batch_size: min batch size for decoding.
        :param max_batch_size: max batch size for decoding.
        :param batch_size_round_frac: bias for rounding.
        :param min_decode_rate: decode until min_decode_rate*max_sample_count
        :param eos_token_id: eos token id
                    None: read from configs.
                    tuple (or list): eos token ids. Set to () to ignore eos.
        :param embedding_dir: embedding path for image embeddings.
        :param spec_algo: speculative decoding algo.
        :param kernels: kernels for attention and MLP.
        :param output_file_name: output file name for results.
        :param output_file_mode: file mode for output file.
        :param output_field_names: output fields.
        :param logger: logger file.
        :param debug: debug or Not.
        """

        assert schedule_mode in ('pingpong', 'mix', 'timely')
        assert output_file_mode in ('w', 'w+', 'a', 'a+')

        assert min_decode_rate > 0.0
        assert max_slot_alloc_fail_count > 0
        if n_stage is None:
            n_stage = torch.cuda.device_count()
        else:
            assert n_stage <= torch.cuda.device_count()

        self.model_path = model_path
        self.model_type, self.torch_dtype, self.n_layer, self.kv_heads, self.head_dim = Reader.get_conf(
            self.model_path)
        self.model_attr = model_attr_map.get(self.model_type, model_attr_map['DEFAULT'])


        self.model_dtype = model_dtype or self.torch_dtype
        self.head_dtype = head_dtype or self.torch_dtype
        self.emb_dtype = emb_dtype or self.torch_dtype
        self.cache_dtype = cache_dtype or self.torch_dtype
        self.n_stage = n_stage
        self.n_proc = n_proc if n_proc else self.n_stage + 1
        self.slot_size = slot_size
        self.schedule_mode = schedule_mode
        self.chunk_size = chunk_size
        self.sync_wait_time = sync_wait_time
        self.queue_timeout = queue_timeout
        self.max_slot_alloc_fail_count = max_slot_alloc_fail_count
        self.slot_first_alloc_rate = slot_first_alloc_rate
        self.slot_fully_alloc_under = slot_fully_alloc_under or 2 ** 16
        self.alloc_early_exit_rate = alloc_early_exit_rate
        self.batch_size_step = batch_size_step
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size_round_frac = batch_size_round_frac
        self.min_decode_rate = min_decode_rate
        self.embedding_dir = embedding_dir
        self.spec_algo = spec_algo
        self.output_file_name = output_file_name
        self.output_file_mode = output_file_mode
        self.output_field_names = output_field_names
        self.logger = logger
        self.debug = debug
        self.kernels = kernels

        self.tokenizer = self.load_tokenizer(self.model_path)

        self.device_list = self.get_device_list(self.n_layer, self.n_stage)
        self.model = self.load_model(self.model_path, self.device_list,
                                     self.n_stage)
        assert self.n_layer == len(
            self.get_module_by_name(self.model, self.model_attr.layer_name))

        assert eos_token_id is None or isinstance(eos_token_id, tuple)
        if eos_token_id is None:
            eos_token_ids = []
            model_eos_token_id = self.model.config.eos_token_id
            if isinstance(model_eos_token_id, (tuple, list)):
                eos_token_ids.extend(list(model_eos_token_id))
            else:
                eos_token_ids.append(model_eos_token_id)
            tokenizer_eos_token_id = self.tokenizer.eos_token_id
            if isinstance(tokenizer_eos_token_id, (tuple, list)):
                eos_token_ids.extend(list(tokenizer_eos_token_id))
            else:
                eos_token_ids.append(tokenizer_eos_token_id)
            eos_token_id = tuple(
                set([x for x in eos_token_ids if x is not None]))
        print(f'eos_token_id:{eos_token_id}')
        self.eos_token_id = eos_token_id

        if cache_size is None:
            mem = torch.cuda.mem_get_info(torch.device(0))[1] / 1e9
            head_size = self.head_dtype.itemsize
            coef = 0.8 + chunk_size / 4096 + 0.05 * head_size  # 0.6->0.8
            cache_size = 1.0 - (2.0 + self.n_proc * coef) / mem

        if cache_size < 1:
            token_size = (2 * self.n_layer * self.kv_heads *
                          self.head_dim * self.cache_dtype.itemsize)
            self.cache_size = self.get_cache_size(self.n_stage, cache_size,
                                                  token_size)
            print(
                f'cache_size:{self.cache_size}/{cache_size:.3f} '
                f'per_proc:{self.cache_size // self.n_proc} '
                f'length(bs=128):{self.cache_size // self.n_proc // 128}')
        else:
            self.cache_size = cache_size
        self.cache = self.init_kv_cache(self.cache_size, self.cache_dtype)

        if self.spec_algo == 'lookahead':
            self.branch_length = 4
            self.max_retrieve_count = 4
            self.spec_buf = self.branch_length * self.max_retrieve_count
            self.spec = Lookahead(table_size=2 ** 24,
                                  branch_count=8*self.max_retrieve_count,
                                  branch_length=self.branch_length,
                                  vocab_size=self.model.vocab_size)
        else:
            self.spec_buf = 0

    def load_model(self, model_path, device_list, n_stage):
        ts = time.time()
        device_map = self.get_device_map(device_list, n_stage)
        from accelerate.hooks import remove_hook_from_module
        Model = model_class_map[self.model_type]
        torch_dtype = self.torch_dtype if (
                self.model_dtype == torch.float8_e4m3fn) else self.model_dtype
        model = Model.from_pretrained(model_path,
                                      torch_dtype=torch_dtype,
                                      low_cpu_mem_usage=True,
                                      device_map=device_map)

        remove_hook_from_module(model, recurse=True)
        # TODO: interleave_value: should be false if use fa3
        patch_kwargs = {"cache_dtype": self.cache_dtype,
                        "interleave_value": self.cache_dtype == torch.float8_e4m3fn,
                        "kernels": self.kernels}
        for name, module in model.named_modules():
            if hasattr(module, 'flood_patch_func'):
                module.flood_patch_func(patch_kwargs)

        if self.head_dtype == torch.float8_e4m3fn:
            print(f"retype lm_head to {self.head_dtype}")
            model.set_output_embeddings(
                model.get_output_embeddings().retype(dtype=torch.float8_e4m3fn))
        if self.emb_dtype == torch.float8_e4m3fn:
            print(f"retype embebdding to {self.emb_dtype}")
            model.set_input_embeddings(
                model.get_input_embeddings().retype(dtype=torch.float8_e4m3fn))

        if self.model_dtype == torch.float8_e4m3fn:
            print(f'retype Linear to torch.float8_e4m3fn')
            for name, module in model.named_modules():
                if isinstance(module, NativeLinear):  # TODO: adapt for MOE
                    self.set_module_by_name(model, name, module.retype(
                        dtype=torch.float8_e4m3fn))

        gc.collect()
        torch.cuda.empty_cache()

        model = model.eval().share_memory()
        # print(model)
        # for name, param in model.named_parameters():
        #     shape = param.data.shape if hasattr(param,'data') else []
        #     print(name, param.dtype, shape, param.device)
        # exit()

        te = time.time()
        print(f'loading time:{te - ts:.1f}s')
        return model

    def load_tokenizer(self, tokenizer_path):
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def get_device_list(self, n_layer, n_stage, counts=None):
        if counts is None:
            if n_layer // n_stage * n_stage == n_layer:
                counts = [n_layer // n_stage] * n_stage 
            else:
                c = (n_layer-1)//n_stage+1
                gap = c*n_stage-n_layer 
                g = gap//2 
                if c-1 > g:
                    counts = [c-1]*g+[c]*(n_stage-gap)+[c-1]*(gap-g)
                else:
                    counts = [c-1]*g+[c]*(n_stage-gap)+[c-1]*(gap-g)
        else:
            assert len(counts) == n_stage and sum(counts) == n_layer
        accums = list(itertools.accumulate([0]+counts))
        indices = [[] for _ in range(n_stage)]
        for i in range(n_stage):
            s = accums[i]
            e = accums[i + 1]
            indices[i].extend(range(s, e))

        for idx, x in enumerate(indices):
            print(f'device:{idx} layer:{x}')
        return indices

    def get_device_map(self, device_list, n_stage):
        device_map = {self.model_attr.emb_name: 0,
                      self.model_attr.norm_name: n_stage - 1,
                      self.model_attr.head_name: n_stage - 1}
        for idx, indices in enumerate(device_list):
            for i in indices:
                device_map[f"{self.model_attr.layer_name}.{i}"] = idx
        return device_map

    def get_cache_size(self, n_stage, max_rate, token_size):
        frees = []
        totals = []
        usages = []
        for i in range(n_stage):
            free_memory, total_memory = torch.cuda.mem_get_info(torch.device(i))
            frees.append(free_memory)
            totals.append(total_memory)
            usages.append(total_memory - free_memory)
        use_str = ' '.join([str(round(x / 2 ** 30, 1)) for x in usages])
        free_str = ' '.join([str(round(x / 2 ** 30, 1)) for x in frees])
        print(f'mem.use:{use_str} \nmem.free:{free_str}')
        cache_size = int((min(totals) * max_rate - max(
            usages)) / token_size * n_stage / 16) * 16
        return cache_size

    def get_module_by_name(self, model, name):
        ns = name.split('.')
        module = model
        for n in ns:
            module = getattr(module, n)
        return module

    def set_module_by_name(self, model, name, module):
        ns = name.split('.')
        if len(ns) == 1:
            setattr(model, name, None)
            setattr(model, name, module)
            return
        prefix = '.'.join(ns[:-1])
        m = self.get_module_by_name(model, prefix)
        key = ns[-1]
        setattr(m, key, None)
        setattr(m, key, module)

    def init_kv_cache(self, max_token, cache_dtype):

        devices = []
        layers = self.get_module_by_name(self.model, self.model_attr.layer_name)
        for i in range(self.n_layer):
            layer = layers[i]
            device = next(layer.parameters()).device
            devices.append(device)
        cache = SegmentCache(max_token,
                             num_layers=self.n_layer,
                             num_key_value_heads=self.kv_heads,
                             head_dim=self.head_dim,
                             dtype=cache_dtype,
                             devices=devices)
        return cache

    def get_sync_layers(self, tasks, device_list):
        update_blocks = []
        for i, indices in enumerate(device_list):
            pre_task = None if i == 0 else tasks[i - 1]
            task = tasks[i]
            sync_layer = TaskSyncLayer(i, pre_task, task, self.sync_wait_time)
            update_blocks.append(sync_layer)
        sync_layer = TaskSyncLayer(-1, tasks[-1], None, self.sync_wait_time)
        update_blocks.append(sync_layer)
        return update_blocks

    def initialize(self, output_queue_count=1):
        input_queue = mp.Queue()
        chunk_quene = mp.Queue()
        working_queue = mp.Queue()
        output_queues = []
        for i in range(output_queue_count):
            output_queues.append(OutputQueue(mp.Queue(), index=i, idle=True))
        return input_queue, chunk_quene, working_queue, output_queues

    def launch(self, input_queue, chunk_quene, working_queue, output_queue):

        torch.cuda.empty_cache()
        gc.collect()

        tasks = [mp.Value('i', 0) for _ in range(self.n_stage)]
        counts = mp.Value('i', 0)
        state = mp.Value('i', 0)
        gbs = mp.Value('i', self.chunk_size)
        allocate_fail_count = mp.Value('i', 0) 

        fail_sample_count = mp.Value('l', 10 ** self.n_proc)

        slots = Array(Slot,
                      [(0, self.cache_size, 1, 0)] + [(0, 0, 0, 0) for i in
                                                      range(self.slot_size)])

        for i in range(self.n_proc):
            process = mp.Process(target=self.schedule,
                                 args=(input_queue,
                                       chunk_quene,
                                       working_queue,
                                       output_queue,
                                       tasks,
                                       slots,
                                       counts,
                                       state,
                                       allocate_fail_count,
                                       fail_sample_count,
                                       gbs,
                                       i),
                                 daemon=True)
            process.start()

        # wait until all processes are ready
        time.sleep(10)

    def schedule(self, input_queue, chunk_quene, working_queue, output_queues,
                 tasks, slots, counts, state, allocate_fail_count,
                 fail_sample_count, gbs, task_id):
        method_name = f'{self.schedule_mode}_schedule'
        return getattr(self, method_name)(input_queue, chunk_quene,
                                          working_queue, output_queues, tasks,
                                          slots, counts, state,
                                          allocate_fail_count,
                                          fail_sample_count, gbs,
                                          task_id)

    def pingpong_schedule(self, input_queue, chunk_quene, working_queue,
                          output_queues, tasks, slots, counts, state,
                          allocate_fail_count, fail_sample_count,
                          gbs, task_id):
        print(
            f"pingpong_schedule task:{task_id} pid:{os.getpid()} ts:{time.time() % 1000:.3f}")
        sync_layers = self.get_sync_layers(tasks, self.device_list)
        device_list = self.device_list
        streams = [torch.cuda.Stream(device=idx) for idx in range(self.n_stage)]
        queue_timeout = self.queue_timeout
        chunk_size = self.chunk_size
        output_queue = output_queues[0].queue

        fails = []
        chunks = []
        waits = []
        options = []
        step = 0
        input_device = torch.device('cpu')
        fe = None
        if self.embedding_dir is not None:
            fe = safe_open(self.embedding_dir, framework="pt", device='cuda:0')
        while True:

            # both empty, wait
            input_empty = input_queue.empty() and fail_sample_count.value == 10 ** self.n_proc  and len(chunks) == 0 and len(options) == 0
            working_empty = working_queue.empty() and len(waits) == 0
            if input_empty and working_empty:
                time.sleep(0.1)
                continue

            task_type = None
            ts = time.time()
            step += 1

            if task_id == 0 and input_empty:
                if self.batch_size_step is None:
                    gbs.value = min(max(2 ** int(
                        math.log2(max(counts.value, 0) / self.n_proc + 1) + 1),
                                        self.min_batch_size),
                                    self.max_batch_size)
                else:
                    gbs.value = min(max(int(max(counts.value,0) / self.n_proc
                                            / self.batch_size_step + 1)
                                        * self.batch_size_step,
                                        self.min_batch_size),
                                    self.max_batch_size)

            if (task_id != 0 and input_empty and
                    counts.value <= self.min_batch_size):
                time.sleep(1.0)
                continue

            dbs = gbs.value

            if  (counts.value < state.value or counts.value == 0
                    and not input_empty):
                state.value = 0
            elif input_empty:
                state.value = self.n_proc * dbs

            # working is near empty, do prefilling
            if state.value == 0:
                n_tokens = 0
                reqs = []
                embs = []
                if len(chunks) > 0:
                    update_chunks = []
                    for req in chunks:
                        n_token = req.input_length - req.done
                        if n_tokens < chunk_size:
                            if n_tokens + n_token <= chunk_size:
                                n_tokens += n_token
                                req.todo = n_token
                            else:
                                todo = chunk_size - n_tokens
                                n_tokens += todo
                                req.todo = todo
                            reqs.append(req)
                            embs.append(self.get_emb(req, fe))
                        else:
                            update_chunks.append(req)
                    chunks = update_chunks

                if len(options) > 0:
                    update_options = []
                    for req in options:
                        # prefill must have be done 
                        gap = req.done - req.input_length 
                        assert gap >= 0
                        target = req.iterate_target()[2]
                        assert target is not None
                        n_token = len(target)
                        if n_tokens < chunk_size:
                            if n_tokens + n_token <= chunk_size:
                                n_tokens += n_token
                                req.todo = n_token
                            else:
                                todo = chunk_size - n_tokens
                                n_tokens += todo
                                req.todo = todo
                            reqs.append(req)
                        else:
                            update_options.append(req)
                    options = update_options
                if n_tokens < chunk_size and len(fails) > 0:
                    update_fails = []
                    for req in fails:
                        assert req.done == 0
                        n_token = req.input_length
                        if n_tokens < chunk_size:
                            if n_tokens + n_token <= chunk_size:
                                n_tokens += n_token
                                req.todo = 0
                            else:
                                todo = chunk_size - n_tokens
                                n_tokens += todo
                                req.todo = todo
                            reqs.append(req)
                            embs.append(self.get_emb(req, fe))
                        else:
                            update_fails.append(req)
                    fails = update_fails

                if n_tokens < chunk_size:
                    while True:
                        try:
                            req = input_queue.get(block=True,
                                                  timeout=queue_timeout)
                        except:
                            break
                        # assert req.done == 0
                        if self.spec_algo == 'lookahead':
                            self.spec.update_state(req.input_ids)
                        n_token = req.input_length
                        if n_tokens + n_token <= chunk_size:
                            n_tokens += n_token
                            req.todo = 0
                            reqs.append(req)
                            embs.append(self.get_emb(req, fe))
                            if n_tokens == chunk_size:
                                break
                        else:
                            todo = chunk_size - n_tokens
                            n_tokens += todo
                            req.todo = todo
                            reqs.append(req)
                            embs.append(self.get_emb(req, fe))
                            break

                if len(reqs) == 0:
                    time.sleep(0.001)
                    continue

                batch = Batch.prefill_batching(reqs, slots,
                                               device=input_device,
                                               min_rate=self.alloc_early_exit_rate,
                                               allocate_rate=self.slot_first_alloc_rate,
                                               fully_alloc_under=self.slot_fully_alloc_under,
                                               cache_size=self.cache_size,
                                               buffer=self.spec_buf,
                                               embeddings=embs)

                if batch.batch_size > 0:
                    task_type = 'prefill'
                    counts.value += sum([x.done == 0 for x in reqs])
                    LLM.update_digit(fail_sample_count, task_id + 1,
                                     len(fails) + len(chunks))
                else:
                    task_type = None
                    # fail to allocate slot, should reput the reqs into inputs
                    sizes = [str(x.todo or x.input_length) + '+' + str(
                        x.output_length) for x in reqs]
                    sizes = ','.join(sizes)
                    slot_stat = f'slots:{Batch.slot_check(slots)}' if self.debug else ''
                    print(
                        f'******** No slots available! task:{task_id} pid:{os.getpid()} dbs:{dbs} ' \
                        f'ips:{input_queue.qsize()} counts:{counts.value} state:{state.value} ' \
                        f'size:{sizes} slots:{slot_stat} ********')
                    for req in reqs:
                        if req.done == 0:
                            fails.append(req)
                        else:
                            chunks.append(req)
                    allocate_fail_count.value += 1

                    LLM.update_digit(fail_sample_count, task_id + 1,
                                     len(fails) + len(chunks))

                    if allocate_fail_count.value >= self.max_slot_alloc_fail_count:
                        if self.batch_size_step is None:
                            gbs_value = min(max(2 ** int(math.log2(
                                counts.value / self.n_proc) + self.batch_size_round_frac),
                                                self.min_batch_size),
                                            self.max_batch_size)
                        else:
                            gbs_value = min(max(int(
                                counts.value / self.n_proc / self.batch_size_step +
                                self.batch_size_round_frac) * self.batch_size_step,
                                                self.min_batch_size),
                                            self.max_batch_size)
                        state.value = min(
                            int(self.min_decode_rate * counts.value),
                            gbs_value * self.n_proc)
                        gbs.value = gbs_value
                        allocate_fail_count.value = 0
                    continue
            else:
                # working is full, do decoding 
                reqs = []
                update_waits = []
                for req in waits:
                    if len(reqs) < dbs:
                        assert req.input_length + len(
                            req.output_ids) >= req.size_of_segs()
                        slot_size = ((req.input_length +
                                      req.output_length - 1  + self.spec_buf) // 16 + 1) * 16
                        extend_length = slot_size - (
                                req.segs[-1][1] - req.segs[-1][0])
                        old_segs = copy.deepcopy(req.segs)
                        segs = Batch.extend_slot(slots, req.segs, extend_length,
                                                 contiguous='sa' not in self.kernels)
                        if segs is not None:
                            print(f'extend success! from {old_segs} to {segs}')
                            req.segs = segs
                            reqs.append(req)
                        else:
                            print(f'extend failed! stay with {req.segs}')
                            update_waits.append(req)
                    else:
                        update_waits.insert(0, req)
                waits = update_waits

                while True:
                    try:
                        req = working_queue.get(block=True,
                                                timeout=queue_timeout)
                    except:
                        break
                    reqs.append(req)
                    if len(reqs) >= dbs:
                        break

                if len(reqs) == 0:
                    time.sleep(0.001)
                    continue

                if any([len(x.segs) > 1 for x in reqs]):
                    pass
                if self.spec_algo == 'lookahead':
                    # TODO: tune the params with different batch size
                    batch = Batch.lookahead_batching(reqs,
                                                     self.spec,
                                                     device=input_device,
                                                     retrieve_length=4,
                                                     retrieve_count=4)
                else:
                    batch = Batch.decoding_batching(reqs, device=input_device)
                task_type = 'decode'

            te = time.time()
            batching_time = te - ts

            ts = te
            self.model.forward(input_ids=batch.input_ids,
                                                    position_ids=batch.position_ids,
                                                    past_key_values=self.cache,
                                                    batch_meta_info=batch,
                                                    device_list=device_list,
                                                    sync_layers=sync_layers,
                                                    streams=streams)

            te = time.time()
            forward_time = te - ts
            ts = te

            for i, req in enumerate(batch.reqs):
                if req.todo > 0 and req.done + req.todo < req.input_length:
                    req.done += req.todo
                    req.todo = 0
                    chunks.append(req)
                else:
                    # reduce pickle cost
                    if task_type == 'prefill':
                        req.input_id = []

                    if req.target_ids is None:
                        if len(req.output_ids) >= req.output_length or req.output_ids[-1] in self.eos_token_id:
                            output_queue.put(req)
                            if self.spec_algo == 'lookahead':
                                self.spec.update_state(req.output_ids)
                            Batch.recycle(slots, req.segs)
                            counts.value -= 1
                        elif req.input_length + len(
                                req.output_ids) >= req.size_of_segs():
                            waits.append(req)
                        else:
                            working_queue.put(req)
                    else:
                        if isinstance(req.target_ids, list):
                            if req.todo == 0: # not chunked
                                req.done = req.input_length 
                                options.append(req)
                            else:  # chunked
                                if req.done + req.todo < req.input_length + sum([len(x) for x in req.target_ids]):  # unfinished
                                    req.done += req.todo
                                    req.todo = 0
                                    options.append(req)
                                else:
                                    output_queue.put(req)
                                    Batch.recycle(slots, req.segs)
                                    counts.value -= 1
                        elif len(req.output_ids) >= 1:
                            output_queue.put(req)
                            Batch.recycle(slots, req.segs)
                            counts.value -= 1
                        else:
                            working_queue.put(req)

            LLM.update_digit(fail_sample_count, task_id + 1,
                             len(fails) + len(chunks))
            te = time.time()
            recycle_time = te - ts

            if self.debug:
                ips = input_queue.qsize()
                wks = working_queue.qsize()
                fail_str = f'{str(fail_sample_count.value)[1:]}'
                tokens = batch.token_count
                bs_str = f'{batch.batch_size}/{tokens}'
                times = (f'{batching_time * 1000:.1f}/'
                         f'{forward_time * 1000:.1f}/{recycle_time * 1000:.1f}')
                print(
                    f'task:{task_id} pid:{os.getpid():<6} step:{step:<4} '
                    f'task_type:{task_type:<7} ' \
                    f'bs:{bs_str:<7} ' \
                    f'counts:{counts.value:<4} state:{state.value:<2} ' \
                    f'ips:{ips} fail:{fail_str}/{len(fails):<2} '
                    f'chunk:{len(chunks):<2} wks:{wks:<4} ' \
                    f'waits:{len(waits):<4} time:{times:<13}')

    def timely_schedule(self, input_queue, chunk_quene, working_queue,
                        output_queues, tasks, slots, counts, state,
                        allocate_fail_count, fail_sample_count,
                        gbs, task_id):
        print(
            f"timely_schedule task:{task_id} pid:{os.getpid()}"
            f" ts:{time.time() % 1000:.3f}")
        sync_layers = self.get_sync_layers(tasks, self.device_list)
        device_list = self.device_list
        streams = [torch.cuda.Stream(device=idx) for idx in range(self.n_stage)]
        queue_timeout = self.queue_timeout
        chunk_size = self.chunk_size

        fails = []
        chunks = []
        waits = []
        step = 0
        input_device = torch.device('cpu')
        pts = 0
        temp_step = 0
        while True:

            # both empty, wait
            input_empty = (input_queue.empty() and
                           fail_sample_count.value == 10 ** self.n_proc)
            working_empty = working_queue.empty() and len(waits) == 0
            if input_empty and working_empty:
                time.sleep(0.001)
                temp_step = 1
                continue
            # chunk_size = max(self.chunk_size-256*temp_step, 512)
            temp_step += 1
            task_type = None
            ts = time.time()

            step += 1

            prefill_gap_time = ts - pts
            counts_value = counts.value
            input_size = input_queue.qsize() + len(chunks) + len(fails)
            min_gap_time = 0.0
            # decay prefill for the last chunk
            if not input_empty and counts_value <= state.value and (
                    input_size > 1 or prefill_gap_time >= min_gap_time):
                state.value = 0
            elif input_empty or input_size == 1 and prefill_gap_time < min_gap_time:
                gbs.value = self.round_batch_size(counts_value,
                                                  self.batch_size_round_frac)
                # upper bound to do prefill asap
                state.value = self.n_proc * self.round_batch_size(counts_value,
                                                                  1.0)
            dbs = gbs.value
            state_value = state.value

            # do prefilling
            if state_value == 0:
                pts = ts
                n_tokens = 0
                reqs = []
                subs = []
                if len(chunks) > 0:
                    update_chunks = []
                    for req in chunks:
                        n_token = req.input_length - req.done
                        if n_tokens < chunk_size:
                            if n_tokens + n_token <= chunk_size:
                                n_tokens += n_token
                                req.todo = n_token
                            else:
                                todo = chunk_size - n_tokens
                                n_tokens += todo
                                req.todo = todo
                                # req should not put into chunks
                                req.task_type = 2
                                subs.append(req)
                            reqs.append(req)
                        else:
                            update_chunks.append(req)
                    chunks = update_chunks

                if n_tokens < chunk_size and len(fails) > 0:
                    update_fails = []
                    for req in fails:
                        assert req.done == 0
                        n_token = req.input_length
                        if n_tokens < chunk_size:
                            if n_tokens + n_token <= chunk_size:
                                n_tokens += n_token
                                req.todo = 0
                            else:
                                todo = chunk_size - n_tokens
                                n_tokens += todo
                                req.todo = todo
                                req.task_type = 2  # req should not put into chunks
                                subs.append(req)
                            reqs.append(req)
                        else:
                            update_fails.append(req)
                    fails = update_fails

                if n_tokens < chunk_size:
                    while True:
                        try:
                            req = input_queue.get(block=True,
                                                  timeout=queue_timeout)
                        except:
                            break
                        # may be subbatch
                        n_token = req.input_length - req.done
                        if n_tokens + n_token <= chunk_size:
                            n_tokens += n_token
                            req.todo = n_token
                            reqs.append(req)
                            if n_tokens == chunk_size:
                                break
                        else:
                            todo = chunk_size - n_tokens
                            n_tokens += todo
                            req.todo = todo
                            reqs.append(req)
                            req.task_type = 2  # req should not put into chunks
                            subs.append(req)
                            break

                if len(reqs) == 0:
                    time.sleep(0.001)
                    continue

                batch = Batch.prefill_batching(reqs, slots, device=input_device,
                                               min_rate=self.alloc_early_exit_rate,
                                               allocate_rate=self.slot_first_alloc_rate,
                                               fully_alloc_under=self.slot_fully_alloc_under,
                                               cache_size=self.cache_size)

                if batch.batch_size > 0:
                    task_type = 'prefill'
                    counts.value += sum([x.done == 0 for x in reqs])
                    LLM.update_digit(fail_sample_count, task_id + 1,
                                     len(fails) + len(chunks))

                    for req in subs:
                        sub_req = copy.deepcopy(req)
                        sub_req.done = req.done + req.todo
                        input_queue.put(sub_req)
                else:
                    task_type = None
                    subs = []
                    # fail to allocate slot, should re-put the reqs into inputs
                    sizes = [str(x.input_length) + '+' + str(x.output_length)
                             for x in reqs]
                    sizes = ','.join(sizes)
                    print(
                        f'No slots available! task:{task_id} pid:{os.getpid()} dbs:{dbs} ' \
                        f'ips:{input_queue.qsize()} counts:{counts.value} state:{state.value} ' \
                        f'size:{sizes}')
                    for req in reqs:
                        if req.done == 0:
                            fails.append(req)
                        else:
                            chunks.append(req)
                    allocate_fail_count.value += 1

                    LLM.update_digit(fail_sample_count, task_id + 1,
                                     len(fails) + len(chunks))

                    if allocate_fail_count.value >= self.max_slot_alloc_fail_count:
                        gbs.value = self.round_batch_size(counts.value,
                                                          self.batch_size_round_frac)
                        # state.value = int(self.min_decode_rate * counts.value)
                        state.value = max(
                            int(self.min_decode_rate * counts.value),
                            counts.value - 16)
                        allocate_fail_count.value = 0
                    continue
            else:
                # do decoding 
                reqs = []
                update_waits = []
                for req in waits:
                    if len(reqs) < min(dbs, 16):
                        assert req.input_length + len(req.output_ids) >= \
                               req.segs[1] - req.segs[0]
                        slot_size = ((req.input_length + req.output_length - 1) // 16 + 1) * 16
                        extend_length = slot_size - (
                                req.segs[-1][1] - req.segs[-1][0])
                        segs = Batch.extend_slot(slots, req.segs, extend_length)
                        if segs is not None:
                            req.segs = segs
                            reqs.append(req)
                        else:
                            update_waits.append(req)
                    else:
                        update_waits.insert(0, req)
                waits = update_waits

                while True:
                    try:
                        req = working_queue.get(block=True,
                                                timeout=queue_timeout)
                    except:
                        break
                    reqs.append(req)
                    if len(reqs) >= dbs:
                        break

                if len(reqs) == 0:
                    time.sleep(0.001)
                    continue

                batch = Batch.decoding_batching(reqs, device=input_device)
                task_type = 'decode'

            te = time.time()
            batching_time = te - ts

            ts = te
            next_token_id_list = self.model.forward(input_ids=batch.input_ids,
                                                    position_ids=batch.position_ids,
                                                    past_key_values=self.cache,
                                                    batch_meta_info=batch,
                                                    device_list=device_list,
                                                    sync_layers=sync_layers,
                                                    streams=streams)

            te = time.time()
            forward_time = te - ts
            ts = te

            for i, next_token_id in enumerate(next_token_id_list):
                req = reqs[i]
                if req.todo > 0 and req.done + req.todo < req.input_length:
                    assert req.task_type == 2
                    # req.done += req.todo
                    # req.todo = 0
                    # chunks.append(req)
                    pass  # subbatch req should be discard
                else:
                    req.output_ids.append(next_token_id)
                    # reduce pickle cost
                    if task_type == 'prefill':
                        req.input_ids = []

                    if (len(req.output_ids) >= req.output_length
                            or next_token_id in self.eos_token_id):
                        req.output_ids.append(
                            None)  # TODO: MUST USE NONE WITH stream MODE
                        output_queues[req.output_index].queue.put(req)
                        Batch.recycle(slots, req.segs)
                        counts.value -= 1
                    elif req.input_length + len(req.output_ids) >= req.segs[1] - \
                            req.segs[0]:
                        waits.append(req)
                    else:
                        if (len(req.output_ids) - 1) % 10 == 0:
                            output_queues[req.output_index].queue.put(req)
                        working_queue.put(req)

            LLM.update_digit(fail_sample_count, task_id + 1,
                             len(fails) + len(chunks))
            te = time.time()
            recycle_time = te - ts

            if self.debug:
                ips = input_queue.qsize()
                wks = working_queue.qsize()
                fail_str = f'{str(fail_sample_count.value)[1:]}'
                tokens = batch.token_count
                bs_str = f'{batch.batch_size}/{tokens}' \
                    if batch.mode == 0 else f'{dbs}/{batch.batch_size}'
                times = f'{batching_time * 1000:.1f}/{forward_time * 1000:.1f}/{recycle_time * 1000:.1f}'
                print(
                    f'task:{task_id} pid:{os.getpid():<6} step:{step:<4} task_type:{task_type:<7} ' \
                    f'bs:{bs_str:<7} ' \
                    f'counts:{counts_value:<4} state:{state_value:<4} ' \
                    f'ips:{ips} fail:{fail_str}/{len(fails):<2} chunk:{len(chunks):<2} wks:{wks:<4} ' \
                    f'waits:{len(waits):<4} time:{times:<13}')

    def mix_schedule(self, input_queue, chunk_quene, working_queue,
                     output_queues, tasks, slots, counts, state,
                     allocate_fail_count, fail_sample_count, gbs,
                     task_id):
        print(
            f"mix_schedule task:{task_id} pid:{os.getpid()} ts:{time.time() % 1000:.3f}")
        sync_layers = self.get_sync_layers(tasks, self.device_list)
        device_list = self.device_list
        streams = [torch.cuda.Stream(device=idx) for idx in range(self.n_stage)]
        queue_timeout = self.queue_timeout
        chunk_size = self.chunk_size
        assert self.slot_first_alloc_rate == 1.0

        output_queue = output_queues[0].queue
        state.value = 4096

        fails = []
        chunks = []
        step = 0
        input_device = torch.device('cpu')
        while True:

            # both empty, wait
            input_empty = (input_queue.empty() and
                           fail_sample_count.value == 10 ** self.n_proc)
            if input_empty and working_queue.empty():
                time.sleep(0.1)
                continue

            task_type = None
            ts = time.time()
            step += 1

            # if task_id == 0 and input_empty:
            #     if self.batch_size_step is None:
            #         gbs.value = min(max(2 ** int(math.log2(counts.value / self.n_proc) + 1),
            #                             self.min_batch_size), self.max_batch_size)
            #     else:
            #         gbs.value = min(max(int(counts.value / self.n_proc / bss + 1) * bss,
            #                             self.min_batch_size), self.max_batch_size)

            if counts.value < self.min_decode_rate * state.value:
                # gbs.value = 2*min(max(2 ** int(math.log2(counts.value / self.n_proc) + 1),
                #                         self.min_batch_size), self.max_batch_size)
                gbs.value = chunk_size
                state.value = 4096

            dbs = gbs.value

            # fill with decoding reqs, if not full, fill with prefill reqs
            reqs = []
            n_tokens = 0

            while True:
                try:
                    req = working_queue.get(block=True, timeout=queue_timeout)
                except:
                    break
                req.task_type = 1
                reqs.append(req)
                n_tokens += 1
                if n_tokens >= dbs:
                    break

            if n_tokens < dbs and counts.value < 0.98 * state.value:

                if len(chunks) > 0:
                    update_chunks = []
                    for req in chunks:
                        n_token = req.input_length - req.done
                        if n_tokens < dbs:
                            if n_tokens + n_token <= dbs:
                                n_tokens += n_token
                                req.todo = n_token
                            else:
                                todo = dbs - n_tokens
                                n_tokens += todo
                                req.todo = todo
                            req.task_type = 0
                            reqs.append(req)
                        else:
                            update_chunks.append(req)
                    chunks = update_chunks

                if n_tokens < dbs and len(fails) > 0:
                    update_fails = []
                    for req in fails:
                        assert req.done == 0
                        n_token = req.input_length
                        if n_tokens < dbs:
                            if n_tokens + n_token <= dbs:
                                n_tokens += n_token
                                req.todo = 0
                            else:
                                todo = dbs - n_tokens
                                n_tokens += todo
                                req.todo = todo
                            req.task_type = 0
                            reqs.append(req)
                        else:
                            update_fails.append(req)
                    fails = update_fails

                if n_tokens < dbs:
                    while True:
                        try:
                            req = input_queue.get(block=False)
                        except:
                            break
                        assert req.done == 0
                        n_token = req.input_length
                        if n_tokens + n_token <= dbs:
                            n_tokens += n_token
                            req.todo = 0
                            req.task_type = 0
                            reqs.append(req)
                            if n_tokens == dbs:
                                break
                        else:
                            todo = dbs - n_tokens
                            n_tokens += todo
                            req.todo = todo
                            req.task_type = 0
                            reqs.append(req)
                            break

            if len(reqs) == 0:
                time.sleep(0.001)
                continue

            batch = Batch.mix_batching(reqs, slots, device=input_device,
                                       min_rate=self.alloc_early_exit_rate)

            if batch.batch_size > 0:
                counts.value += sum(
                    [x.task_type == 0 and x.done == 0 for x in reqs])
                LLM.update_digit(fail_sample_count, task_id + 1,
                                 len(fails) + len(chunks))
            else:
                # fail to allocate slot, should cache the reqs into fails
                sizes = [str(x.input_length) + '+' + str(x.output_length) for x
                         in reqs if x.task_type == 0 and x.done == 0]
                sizes = ','.join(sizes)
                print(f'No slots available! task:{task_id} pid:{os.getpid()} '
                      f'ips:{input_queue.qsize()} counts:{counts.value} state:{state.value} '
                      f'size:{sizes}')
                for req in reqs:
                    if req.task_type == 1:
                        working_queue.put(req)
                    elif req.done == 0:
                        fails.append(req)
                    else:
                        chunks.append(req)
                allocate_fail_count.value += 1

                LLM.update_digit(fail_sample_count, task_id + 1,
                                 len(fails) + len(chunks))

                if allocate_fail_count.value >= self.max_slot_alloc_fail_count:
                    if self.batch_size_step is None:
                        dbs_value = min(
                            max(2 ** int(math.log2(
                                counts.value / self.n_proc) + self.batch_size_round_frac),
                                self.min_batch_size), self.max_batch_size)
                    else:
                        dbs_value = min(
                            max(int(
                                counts.value / self.n_proc / self.batch_size_step +
                                self.batch_size_round_frac) * self.batch_size_step,
                                self.min_batch_size), self.max_batch_size)
                    gbs.value = dbs_value
                    # state.value = dbs_value * self.n_proc
                    state.value = counts.value
                    allocate_fail_count.value = 0
                continue

            te = time.time()
            batching_time = te - ts

            ts = te
            next_token_id_list = self.model.forward(input_ids=batch.input_ids,
                                                    position_ids=batch.position_ids,
                                                    past_key_values=self.cache,
                                                    batch_meta_info=batch,
                                                    device_list=device_list,
                                                    sync_layers=sync_layers,
                                                    streams=streams)

            for i, next_token_id in enumerate(next_token_id_list):
                req = reqs[i]
                # TODO: allocate case may cause bug
                if (req.task_type == 0 and req.todo > 0
                        and req.done + req.todo < req.input_length):
                    req.done += req.todo
                    req.todo = 0
                    chunks.append(req)
                else:
                    req.output_ids.append(next_token_id)
                    # reduce pickle cost
                    if req.task_type == 1:
                        req.input_ids.clear()

                    if (len(req.output_ids) >= req.output_length
                            or next_token_id in self.eos_token_id):
                        output_queue.put(req)
                        Batch.recycle(slots, req.segs)
                        counts.value -= 1
                    else:
                        working_queue.put(req)

            # slot is not enough, recomputing samples are put into fails
            LLM.update_digit(fail_sample_count, task_id + 1,
                             len(fails) + len(chunks))

            if self.debug:
                ips = input_queue.qsize()
                wks = working_queue.qsize()
                fail_str = f'{str(fail_sample_count.value)[1:]}'
                decode_tokens = sum([x.task_type for x in reqs])
                prefill_tokens = n_tokens - decode_tokens
                bs_str = f'{n_tokens}/{prefill_tokens}/{decode_tokens}'

                print(f'pid:{os.getpid()} step:{step:<4} ' \
                      f'bs:{bs_str:<11} ' \
                      f'counts:{counts.value:<4} state:{state.value:<2} ' \
                      f'ips:{ips} fail:{fail_str}/{len(fails):<2} wks:{wks:<4} ' \
                      f'batch:{batching_time * 1000:<4.1f}ms '
                      f'forward:{(time.time() - ts) * 1000:<5.1f}ms')

    def generate(self, requests: List, input_queue, output_queues,
                 print_count=0, log_info=''):
        responses = []
        for response in self.request_stream_generate(requests, input_queue,
                                             output_queues,
                                             print_count=print_count,
                                             log_info=log_info):
            responses.append(response)
        return responses

    def request_stream_generate(self, requests: List, input_queue, output_queues,
                        print_count=0, log_info=''):
        logger = open(self.logger, 'a+')
        params = self.format_params(len(requests), log_info=log_info)
        print(params)
        logger.write(params)
        logger.flush()

        tokenizer = self.tokenizer

        handler = self.before_process()
        output_queue = output_queues[0].queue

        n_sample = len(requests)
        req_dict = OrderedDict()
        for req in requests:
            req_dict[req.rid] = req

        ts = time.time()

        total_slot_size = 0
        for req in requests:
            input_ids = tokenizer(req.input_text, add_special_tokens=False)[
                'input_ids']
            req.input_ids = input_ids
            req.input_length = len(input_ids)
            output_slot_size = req.output_length \
                if req.output_length <= self.slot_fully_alloc_under \
                else self.slot_first_alloc_rate * req.output_length
            slot_size = ((req.input_length + int(
                output_slot_size) - 1) // 16 + 1) * 16
            r = Req(req.rid, input_ids=input_ids, input_length=req.input_length,
                    output_length=req.output_length,
                    emb_idx=req.emb_idx,
                    emb_size=req.emb_size,
                    target_ids=req.target_ids,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    min_p=req.min_p)
            input_queue.put(r, block=True)
            total_slot_size += slot_size
        te = time.time()
        print(
            f'pid:{os.getpid()} tokenize:{te - ts:.3f}s '
            f'ts:{time.time() % 1000:.3f} '
            f'mean_slot_size:{total_slot_size / n_sample:.0f}')

        steps = [ts]
        batch_token_counts = []
        responses = []
        output_sample_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        while output_sample_count < n_sample:
            req = output_queue.get(block=True)
            request = req_dict[req.rid]

            output_ids = req.output_ids
            total_input_tokens += request.input_length
            total_output_tokens += len(output_ids)

            if output_ids[-1] in self.eos_token_id:
                output_ids = output_ids[:-1]
            if len(output_ids) > 0 and isinstance(output_ids[0], list):
                target_tokens = request.target_tokens
                output_text = ' '.join(
                    [target_tokens[i] + ":" + str(x) for i, x in
                     enumerate(output_ids[0])])
            else:
                output_text = tokenizer.decode(output_ids)

            request.output_text = output_text
            request.output_ids = output_ids
            responses.append(request)
            output_sample_count += 1
            self.process_requests(handler, [request])

            yield request

            if output_sample_count <= print_count:
                fmt_output_text = output_text.replace('\n', ' ')
                print(
                    f"\n------------ {output_sample_count}  -------------")
                print(f"**** prompt ****:{request.input_text[-100:]}")
                # print((f"**** input_ids ****:{request.input_ids}")
                print(f"**** answer ****:{fmt_output_text}")
                # print(f"**** output_ids ****:{request.output_ids}\n")

            if output_sample_count % 100 == 0:
                steps.append(time.time())
                batch_time = steps[-1] - steps[-2]
                batch_tokens = sum(
                    [len(x.output_ids) for x in responses[-100:]])
                batch_token_counts.append(batch_tokens)
                speed = batch_tokens / batch_time

                total_time = steps[-1] - steps[0]
                accum_speed = total_output_tokens / total_time

                slide_time = steps[-1] - steps[-min(10, len(steps))]
                slide_tokens = sum(batch_token_counts[-10:])
                slide_speed = slide_tokens / slide_time

                log(logger,
                    f'sample:{output_sample_count} time:{batch_time:.3f}s '
                    f'speed:{speed:.2f}token/s ' \
                    f'slide:{slide_speed:.2f}token/s '
                    f'accum:{accum_speed:.2f}token/s')

        self.after_process(handler)
        te = time.time()
        elapse = te - ts
        mean_input_length = total_input_tokens / output_sample_count
        mean_output_length = total_output_tokens / output_sample_count
        log(logger,
            f'\nsample:{output_sample_count} time:{elapse:.2f}s '
            f'input_token:{mean_input_length:.0f}/{total_input_tokens} ' \
            f'output_token:{mean_output_length:.0f}/{total_output_tokens} ' \
            f'throughput:{total_output_tokens / elapse:.2f}token/s\n')
        logger.close()

    def tokenize(self, requests, input_queue, qps=None):

        ts = time.time()
        for req in requests:
            receive_ts = time.time()
            input_ids = self.tokenizer(req.input_text)['input_ids']
            req.input_ids = input_ids
            req.input_length = len(input_ids)
            r = Req(req.rid, input_ids=input_ids, input_length=req.input_length,
                    output_length=req.output_length, emb_idx=req.emb_idx,
                    emb_size=req.emb_size)
            input_queue.put(r, block=True)
            if qps is not None:
                time.sleep(1.0 / qps)
        te = time.time()
        print(
            f'pid:{os.getpid()} tokenize:{te - ts:.3f}s ts:{time.time() % 1000:.3f}')

    async def async_stream_generate(self, request: Request, input_queue,
                                    output_queue):

        tokenizer = self.tokenizer

        input_ids = tokenizer(request.input_text)['input_ids']
        request.input_ids = input_ids
        output_length = request.output_length
        input_length = len(input_ids)
        request.input_length = input_length

        req = Req(request.rid, input_ids=input_ids, input_length=input_length,
                  output_length=output_length, stream=True,
                  output_index=request.output_index)

        input_queue.put(req)

        cursor = 0
        while True:
            try:
                req = output_queue.get(block=False)
            except:
                await asyncio.sleep(0.01)
                continue
            output_ids = req.output_ids[cursor:]
            finished = True if output_ids[-1] is None else False
            if finished:
                output_ids = output_ids[:-1]
            if output_ids[-1] in self.eos_token_id:
                output_ids = output_ids[:-1]
            output_text = tokenizer.decode(output_ids)
            yield {"text": output_text, "token_count": len(output_ids)}
            if finished:
                break
            cursor += len(output_ids)

    def round_batch_size(self, count_value, frac=0.0):
        if self.batch_size_step is None:
            value = min(max(2 ** int(
                math.log2(max(count_value / self.n_proc, 1)) + frac),
                            self.min_batch_size), self.max_batch_size)
        else:
            value = min(max(int(max(count_value / self.n_proc,
                                    1) / self.batch_size_step + frac) * self.batch_size_step,
                            self.min_batch_size), self.max_batch_size)
        return value

    def format_params(self, n_sample, log_info=''):
        fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        device = str(self.n_stage) + '*' + \
                 torch.cuda.get_device_name().split(' ')[-1]
        params = f'\ntime:{fmt} ' \
                 f'model:{self.model_path}  ' \
                 f'torch_dtype:{self.torch_dtype} ' \
                 f'model_dtype:{self.model_dtype}  ' \
                 f'cache_dtype:{self.cache_dtype} ' \
                 f'head_dtype:{self.head_dtype} '\
                 f'emb_dtype:{self.emb_dtype} ' \
                 f'n_layer:{self.n_layer} ' \
                 f'n_stage:{self.n_stage} ' \
                 f'n_proc:{self.n_proc} ' \
                 f'cache_size:{self.cache_size} ' \
                 f'slot_size:{self.slot_size} ' \
                 f'schedule_mode:{self.schedule_mode} ' \
                 f'chunk_size:{self.chunk_size} ' \
                 f'sync_wait_time:{self.sync_wait_time} ' \
                 f'queue_timeout:{self.queue_timeout} ' \
                 f'max_slot_alloc_fail_count:{self.max_slot_alloc_fail_count} ' \
                 f'batch_size_step:{self.batch_size_step} '\
                 f'min_batch_size:{self.min_batch_size} ' \
                 f'max_batch_size:{self.max_batch_size} '\
                 f'slot_first_alloc_rate:{self.slot_first_alloc_rate} ' \
                 f'slot_fully_alloc_under:{self.slot_fully_alloc_under} ' \
                 f'batch_size_round_frac:{self.batch_size_round_frac} ' \
                 f'alloc_early_exit_rate:{self.alloc_early_exit_rate} ' \
                 f'min_decode_rate:{self.min_decode_rate} ' \
                 f'kernels:{self.kernels} ' \
                 f'spec_algo:{self.spec_algo} ' \
                 f'debug:{self.debug} ' \
                 f'eos:{self.eos_token_id} ' \
                 f'sample:{n_sample} ' \
                 f'device:{device} ' \
                 f'{log_info}\n'
        return params

    def get_emb(self, req, fe):
        if req.emb_size == 0:
            return None
        if req.todo == 0:
            return fe.get_tensor(req.rid)
        if (req.done >= req.emb_idx + req.emb_size or
                req.done + req.todo <= req.emb_idx):
            return None
        return fe.get_tensor(req.rid)

    def before_process(self):
        handler = None
        if self.output_file_name is not None:
            handler = open(self.output_file_name, self.output_file_mode)
        return handler

    def process_requests(self, handler, requests):
        if handler is None:
            return
        for request in requests:
            kvs = request.__dict__
            if self.output_field_names is None:
                handler.write(json.dumps(kvs, ensure_ascii=False) + '\n')
            else:
                dumps = {}
                keys = ('rid',) + self.output_field_names
                for k, v in kvs.items():
                    if k in keys:
                        dumps[k] = v
                handler.write(json.dumps(dumps, ensure_ascii=False) + '\n')

    def after_process(self, handler):
        if handler is not None:
            handler.close()

    @staticmethod
    def log_mem():
        free_memory, total_memory = torch.cuda.mem_get_info(torch.device(i))
        used_memory = total_memory - free_memory
        print(
            f"pid:{os.getpid() % 100} used:{used_memory / 2 ** 30:.2f}GiB "
            f"free:{free_memory / 2 ** 30:.2f}GiB\n")

    @staticmethod
    def update_digit(share_value, index, value):
        vals = list(str(share_value.value))
        vals[index] = str(min(value, 1))
        share_value.value = int(''.join(vals))
