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
        
        if self.model_type =='DeepseekV3ForCausalLM':
            dims = [self.model.config.kv_lora_rank+self.model.config.qk_rope_head_dim]  
        else:
            dims = [self.kv_heads*self.head_dim]*2
        if self.fix_size_indices is  None:
            fix_size_dim = None
        else:
            fix_size_dim = self.conf['num_attention_heads']*self.head_dim**2
        cache = SegmentCache(max_token,
                            num_layers=self.n_layer,
                            dims=dims,
                            max_concurrency=self.max_concurrency,
                            fix_size_dim=fix_size_dim,
                            dtype=cache_dtype,
                            devices=devices,
                            fix_size_indices=self.fix_size_indices,
                            )
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
        batching_stream = torch.cuda.Stream(device=0)
        streams = [torch.cuda.Stream(device=idx) for idx in range(self.n_stage)]
        queue_timeout = self.queue_timeout
        chunk_size = self.chunk_size
        output_queue = output_queues[0].queue
        fully_alloc_under = self.slot_fully_alloc_under

        fe = None
        if self.embedding_dir is not None:
            fe = safe_open(self.embedding_dir, framework="pt", device='cuda:0')
        input_lengths = []  # input_length of finished reqs
        output_lengths = []  #  output_length of finished reqs
        fails = []
        chunks = []
        waits = []
        options = []
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
            input_empty = input_queue.empty() and fail_sample_count.value == 10 ** self.n_proc and len(chunks) == 0 and len(options) == 0
            working_empty = working_queue.empty() and len(waits) == 0 and counts.value == 0  # TODO: CHECK counts.value
            if input_empty and working_empty:
                time.sleep(0.001)
                continue

            task_type = None
            ts = time.time()
            step += 1

            if task_id == 0:
                gbs.value = self.opt_batch_size(counts.value, self.n_proc)

            hungry = counts.value <= self.n_proc * self.min_batch_size

            if (task_id != 0 and input_empty and
                    counts.value <= self.min_batch_size and self.spec_algo is None):
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
                        if self.spec_algo == 'lookahead' and hungry:
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

                if len(output_lengths) >= 512 and self.tune_alloc_size:
                    fully_alloc_under = sorted(output_lengths)[int(0.9*len(output_lengths))]

                batch = Batch.prefill_batching(reqs, slots,
                                               fix_slots=fix_slots,
                                               device=input_device,
                                               min_rate=self.alloc_early_exit_rate,
                                               fully_alloc_under=fully_alloc_under,
                                               cache_size=self.cache_size,
                                               buffer_size=self.spec_buf,
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
                        gbs.value = self.opt_batch_size(counts.value, self.n_proc)
                        if input_queue.qsize() < 16 and state.value > 32:
                            state.value -= 32
                        else:
                            state.value = min(
                                int(self.min_decode_rate * counts.value),
                                gbs.value * self.n_proc)
                        allocate_fail_count.value = 0
                    continue
            else:
                # working is full, do decoding 
                reqs = []
                update_waits = []
                n_suc = 0
                n_fail = 0
                for req in waits:
                    if len(reqs) < dbs:
                        size_of_segs = req.size_of_segs()
                        assert req.input_length + len(
                            req.output_ids) == size_of_segs + 1
                        slot_size = req.input_length + req.output_length + self.spec_buf
                        extend_length = min(slot_size - size_of_segs, self.max_extend_size)
                        slot_size = ((extend_length - 1) // 128 + 1) * 128
                        segs = Batch.extend_slot(slots, 
                                                 req.segs, 
                                                 extend_length,
                                                 contiguous='sa' not in self.kernels)
                        if segs is not None:
                            # print(f'extend success! from {req.slot} to {slot}')
                            n_suc += 1
                            req.segs = segs
                            reqs.append(req)
                        else:
                            # print(f'extend {req.segs} failed!')
                            if 'sa' in self.kernels:  # no segment available
                                allocate_fail_count.value += 1

                                if allocate_fail_count.value >= self.max_slot_alloc_fail_count:
                                    gbs.value = self.opt_batch_size(counts.value//2, self.n_proc)
                                    state.value = min(
                                        int(self.min_decode_rate * counts.value),
                                        gbs.value * self.n_proc)
                                    allocate_fail_count.value = 0
                            n_fail += 1
                            update_waits.append(req)
                    else:
                        update_waits.insert(0, req)
                if len(waits) > 0:
                    print(f'extend slot: suc:{n_suc} fail:{n_fail}')
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

                if self.spec_algo == 'lookahead' and hungry:
                    min_buf = min([x.size_of_segs() - x.input_length - len(x.output_ids) for x in reqs])
                    if min_buf >= self.spec_branch_length:
                        max_retrieve_count = min(min_buf//self.spec_branch_length, self.max_spec_branch_count)
                        retrieve_count = min(max(64*self.n_proc//max(self.spec_branch_length*counts.value,1), 1), max_retrieve_count)
                        with torch.cuda.stream(batching_stream):
                            batch = Batch.lookahead_batching(reqs,
                                                            self.spec,
                                                            device=input_device,
                                                            retrieve_count=retrieve_count)
                            batching_stream.synchronize()
                        task_type = 'spec'
                    else:
                        batch = Batch.decoding_batching(reqs, device=input_device)
                        task_type = 'decode'  
                else:
                    batch = Batch.decoding_batching(reqs, device=input_device)
                    task_type = 'decode'

            te = time.time()
            batching_time = te - ts
            step += 1
            ts = te
            hidden_states = self.model.forward(input_ids=batch.input_ids,
                                                    position_ids=batch.position_ids,
                                                    past_key_values=self.cache,
                                                    batch_meta_info=batch,
                                                    device_list=device_list,
                                                    sync_layers=sync_layers,
                                                    streams=streams)
            te = time.time()
            forward_time = te - ts
            ts = te
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
                        if req.target_ids is None:
                             # remove token_ids after eos
                            if task_type == 'spec':
                                for j in range(max(-self.spec_branch_length-1,-len(req.output_ids)),-1):
                                    if req.output_ids[j] in self.eos_token_id:
                                        req.output_ids = req.output_ids[:j+1]
                                        break
                            if len(req.output_ids) >= req.output_length or req.output_ids[-1] in self.eos_token_id:
                                output_queue.put(req)
                                input_lengths.append(req.input_length)
                                output_lengths.append(len(req.output_ids))

                                if len(input_lengths) >= 2048:
                                    input_lengths = input_lengths[-1024:]
                                    output_lengths = output_lengths[-1024:]
                                Batch.recycle(slots, req.segs, fix_slots=fix_slots, fix_size_slot_index=req.fix_size_slot_index)
                                if self.spec_algo == 'lookahead':
                                    self.spec.update_state(req.output_ids)
                                counts.value -= 1
                            elif req.input_length + len(
                                    req.output_ids) >= req.size_of_segs():
                                print(
                                    f'{req.input_length=} {len(req.output_ids)=} {req.output_length=} {req.segs=}')
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
                                        Batch.recycle(slots, req.segs, fix_slots, req.fix_size_slot_index)
                                        counts.value -= 1
                                        if isinstance(req.target_ids[0], list):
                                            for i, target_id in enumerate(req.target_ids):
                                                req.output_ids[0][i] /= len(target_id)
                            elif len(req.output_ids) >= 1:
                                output_queue.put(req)
                                Batch.recycle(slots, req.segs, fix_slots, req.fix_size_slot_index)
                                counts.value -= 1
                            else:
                                working_queue.put(req)
                LLM.update_digit(fail_sample_count, task_id + 1, len(fails) + len(chunks))
                te = time.time()
                remote_time = te - ts

                if self.debug:
                    ips = input_queue.qsize()
                    wks = working_queue.qsize()
                    fail_str = f'{str(fail_sample_count.value)[1:]}'
                    tokens = batch.token_count
                    bs_str = f'{batch.batch_size}/{tokens}'
                    times = (f'{batching_time * 1000:.1f}/'
                            f'{forward_time * 1000:.1f}/{remote_time * 1000:.1f}')
                    mean_input_length = sum(input_lengths)/max(len(input_lengths),1)
                    mean_output_length = sum(output_lengths)/max(len(output_lengths),1)
                    print(
                        f'task:{task_id} step:{step:<4} '
                        f'task_type:{task_type:<7} ' \
                        f'bs:{bs_str:<7} ' \
                        f'counts:{counts.value:<4} state:{state.value:<2} ' \
                        f'ips:{ips} fail:{fail_str}/{len(fails):<2} '
                        f'chunk:{len(chunks):<2} wks:{wks:<4} ' \
                        f'waits:{len(waits):<4} ' \
                        f'length:{mean_input_length:.0f}/{mean_output_length:.0f}/{fully_alloc_under} ' \
                        f'time:{times:<13}')
