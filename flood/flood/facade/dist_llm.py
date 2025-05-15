# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import gc
import math
import os
import time

import torch
import torch.distributed as dist
from safetensors import safe_open

from flood.facade.llm import LLM
from flood.layers.linear import NativeLinear
from flood.utils.batch import Batch
from flood.utils.cache import SegmentCache
from flood.models import model_class_map


class DistLLM(LLM):
    def __init__(self, *args, **kwargs):
        # rename RANK with prefix FLOOD to run within dist docker
        self.rank = int(os.environ["FLOOD_RANK"])
        self.world_size = int(os.environ["FLOOD_WORLD_SIZE"])
        self.master = os.environ['FLOOD_MASTER']
        self.port = int(os.environ['FLOOD_PORT'])
        super().__init__(*args, **kwargs)

    def load_model(self, model_path, device_list, n_stage):
        ts = time.time()
        device_map = self.get_device_map(device_list, n_stage)
        from accelerate.hooks import remove_hook_from_module
        Model = model_class_map[self.model_type]
        torch_dtype = self.torch_dtype if self.model_dtype == torch.float8_e4m3fn else self.model_dtype
        model = Model.from_pretrained(model_path,
                                      torch_dtype=torch_dtype,
                                      low_cpu_mem_usage=True,
                                      device_map=device_map)
        remove_hook_from_module(model, recurse=True)
        # TODO: interleave_value should be false if use fa3
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
                    # print(f'retype {name} to torch.float8_e4m3fn')
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

    def get_device_list(self, n_layer, n_stage, counts=None):
        assert n_layer // (
                    n_stage * self.world_size) * n_stage * self.world_size == n_layer
        if counts is None:
            counts = [n_layer // n_stage // self.world_size] * (
                        n_stage * self.world_size)
        else:
            assert len(counts) == n_stage and sum(counts) == n_layer
        accums = [0]
        for c in counts:
            accum = accums[-1] + c
            accums.append(accum)

        indices = [[] for _ in range(n_stage)]
        for i in range(n_stage):
            s = accums[i + self.rank * n_stage]
            e = accums[i + self.rank * n_stage + 1]
            indices[i].extend(range(s, e))

        for idx, x in enumerate(indices):
            print(f'node:{self.rank} device:{idx} layer:{x}')
        return indices

    def get_cache_size(self, n_stage, max_rate, token_size, device_list):
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
            usages)) / token_size * n_stage / self.world_size / 16) * 16
        return cache_size

    def init_kv_cache(self, max_token, cache_dtype):

        devices = []
        layers = self.get_module_by_name(self.model, self.model_attr.layer_name)
        for i in range(self.n_layer):
            layer = layers[i]
            if layer.layer_idx is None:
                device = None
            else:
                device = next(layer.parameters()).device
            devices.append(device)
        cache = SegmentCache(max_token,
                             num_layers=self.n_layer,
                            #  num_key_value_heads=self.kv_heads,
                            #  head_dim=self.head_dim,
                             dtype=cache_dtype,
                             devices=devices)
        return cache

    def pingpong_schedule(self, input_queue, chunk_quene, working_queue,
                          output_queues, tasks, slots, fix_slots, counts, state, allocate_fail_count, fail_sample_count,
                          gbs, task_id):
        print(
            f"schedule:pingpong rank:{self.rank} task:{task_id} pid:{os.getpid()} ts:{time.time() % 1000:.3f}")

        http = f'tcp://{self.master}:{self.port + task_id}'
        dist.init_process_group(backend='nccl', init_method=http,
                                world_size=self.world_size, rank=self.rank)
        group = None

        sync_layers = self.get_sync_layers(tasks, self.device_list)
        device_list = self.device_list
        streams = [torch.cuda.Stream(device=idx) for idx in range(self.n_stage)]
        queue_timeout = self.queue_timeout
        chunk_size = self.chunk_size
        output_queue = output_queues[0].queue

        fe = None
        if self.embedding_dir is not None:
            fe = safe_open(self.embedding_dir, framework="pt", device='cuda:0')
        fails = []
        chunks = []
        waits = []
        step = 0
        input_device = torch.device('cpu')
        while True:
            comm_device = None
            # only support two nodes currently
            if self.rank > 0:
                batch = Batch()
                hidden_states = batch.recv(src=self.rank - 1, group=group)
                ts_ = time.time()
                hidden_states = self.model.forward(
                    input_ids=batch.input_ids,
                    hidden_states=hidden_states,
                    position_ids=batch.position_ids,
                    past_key_values=self.cache,
                    batch_meta_info=batch,
                    device_list=device_list,
                    sync_layers=sync_layers,
                    streams=streams)
                te_ = time.time()
                if self.debug:
                    modes = 'prefill' if batch.mode == 0 else 'decode'
                    print(
                        f'rank:{self.rank} task:{task_id} {modes} ' \
                        f'token:{batch.token_count} time:{(te_ - ts_) * 1000:.2f}ms')

                if self.rank == self.world_size - 1:
                    dist.send_object_list(
                        [[batch.mode, batch.reqs]], dst=0,
                        group=group, device=comm_device)
                continue

            # both empty, wait
            input_empty = input_queue.empty() and fail_sample_count.value == 10 ** self.n_proc
            working_empty = working_queue.empty() and len(waits) == 0
            if input_empty and working_empty:
                time.sleep(0.001)
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
                    gbs.value = min(max(int(max(counts.value,
                                                0) / self.n_proc / self.batch_size_step + 1) * self.batch_size_step,
                                        self.min_batch_size),
                                    self.max_batch_size)

            if task_id != 0 and input_empty and counts.value <= self.min_batch_size:
                time.sleep(0.001)
                continue

            dbs = gbs.value

            if (counts.value < state.value or counts.value == 0) and not input_empty:
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
                                               fully_alloc_under=self.slot_fully_alloc_under,
                                               cache_size=self.cache_size,
                                               embeddings=embs)

                if batch.batch_size > 0:
                    task_type = 'prefill'
                    counts.value += sum([x.done == 0 for x in reqs])
                    LLM.update_digit(fail_sample_count, task_id + 1, len(fails) + len(chunks))
                else:
                    task_type = None
                    # fail to allocate slot, should reput the reqs into inputs
                    sizes = [str(x.todo or x.input_length) + '+' + str(
                        x.output_length) for x in reqs]
                    sizes = ','.join(sizes)
                    print(
                        f'******** No slots available! task:{task_id} pid:{os.getpid()} dbs:{dbs} ' \
                        f'ips:{input_queue.qsize()} counts:{counts.value} state:{state.value} ' \
                        f'size:{sizes} slots:{Batch.slot_check(slots)} ********')
                    for req in reqs:
                        if req.done == 0:
                            fails.append(req)
                        else:
                            chunks.append(req)
                    allocate_fail_count.value += 1

                    LLM.update_digit(fail_sample_count, task_id + 1, len(fails) + len(chunks))

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
                # dbs = min(( (counts.value-1) // self.n_proc // 16 + 1)*16, dbs)
                update_waits = []
                for req in waits:
                    if len(reqs) < dbs:
                        assert req.input_length + len(req.output_ids) >= \
                               req.segs[1] - req.segs[0]
                        slot_size = ((req.input_length + req.output_length - 1) // 16 + 1) * 16
                        segs = Batch.extend_slot(slots, req.segs, (
                        req.segs[0], req.segs[0] + slot_size, req.segs[2]))
                        if segs is not None:
                            # print(f'extend success! from {req.slot} to {slot}')
                            req.segs = segs
                            reqs.append(req)
                        else:
                            # print(f'extend failed! {req.slot}')
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

            ts = time.time()
            hidden_states = self.model.forward(input_ids=batch.input_ids,
                                                    position_ids=batch.position_ids,
                                                    past_key_values=self.cache,
                                                    batch_meta_info=batch,
                                                    device_list=device_list,
                                                    sync_layers=sync_layers,
                                                    streams=streams)
            te = time.time()
            forward_time = te - ts
            ts = time.time()
            if self.rank < self.world_size - 1:
                batch.send(hidden_states, dst=self.rank + 1, group=group)

            if self.rank == 0:
                objects = [None]
                dist.recv_object_list(objects, src=self.world_size - 1,
                                      group=group, device=comm_device)
                objects = objects[0]
                mode = objects[0]
                reqs = objects[1]
                for i, req in enumerate(reqs):
                    if req.todo > 0 and req.done + req.todo < req.input_length:
                        req.done += req.todo
                        req.todo = 0
                        chunks.append(req)
                    else:
                         # reduce pickle cost
                        if task_type == 'prefill':
                            req.input_id = []

                        if len(req.output_ids) >= req.output_length or req.output_ids[-1] in self.eos_token_id:
                            output_queue.put(req)
                            Batch.recycle(slots, req.segs)
                            counts.value -= 1
                        elif req.input_length + len(
                                req.output_ids) >= req.size_of_segs():
                            print(
                                f'{req.input_length=} {len(req.output_ids)=} {req.output_length=} {req.segs=}')
                            waits.append(req)
                        else:
                            working_queue.put(req)
                LLM.update_digit(fail_sample_count, task_id + 1, len(fails) + len(chunks))
                te = time.time()
                remote_time = te - ts

                if self.debug:
                    ips = input_queue.qsize()
                    wks = working_queue.qsize()
                    fail_str = f'{str(fail_sample_count.value)[1:]}'
                    bs_str = f'{batch.token_count}/{batch.batch_size}' if mode == 0 else f'{dbs}/{batch.batch_size}'
                    times = f'{batching_time * 1000:.1f}/{forward_time * 1000:.1f}/{remote_time * 1000:.1f}'
                    task_type = 'prefill' if mode == 0 else 'decode'
                    print(
                        f'task:{task_id} pid:{os.getpid():<6} step:{step:<4} task_type:{task_type:<7} ' \
                        f'bs:{bs_str:<7} ' \
                        f'counts:{counts.value:<4} state:{state.value:<2} ' \
                        f'ips:{ips} fail:{fail_str}/{len(fails):<2} chunk:{len(chunks):<2} wks:{wks:<4} ' \
                        f'waits:{len(waits):<4} time:{times:<13}')
