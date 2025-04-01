# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import itertools
import math
from ctypes import Structure, c_int

import numpy as np
import torch
import torch.distributed as dist
from flood.ops.draft import retrieve_draft_table

from flood.utils.sampling import Sampling

"""
    meta.batch_size = bs
    meta.q_offsets = q_offsets
    meta.k_offsets = k_offsets
    meta.q_lengths = q_lengths
    meta.k_lengths = k_lengths
    meta.k_segs = k_segs
    meta.max_seqlen_q = max_q_length
    meta.max_seqlen_k = max_k_length
    meta.max_seg = max_seg
    meta.mask = mask
    meta.qls = qls
    meta.kls = klss
"""

"""
state=0: undefined
state=1: available
state=2: occupied
state=3: available and reserved
state=4: occupied and reserved

share=0: private segment
share=1: prefix cache
"""


class Slot(Structure):
    _fields_ = [('s', c_int), ('e', c_int), ('state', c_int), ('share', c_int)]


class Batch:
    def __init__(self,
                 batch_size=0,
                 token_count=0,
                 mode=1,
                 samplings=None,
                 input_ids=None,
                 position_ids=None,
                 pids=None,
                 q_offsets=None,
                 k_offsets=None,
                 max_q_length=None,
                 max_k_length=None,
                 k_segs=None,
                 max_seg=None,
                 q_lengths=None,
                 k_lengths=None,
                 mask=None,
                 cache_indices=None,
                 logit_indices=None,
                 embeddings=None,
                 emb_idx_list=None,
                 reqs=None,
                 qls=None,
                 kls=None,
                 spec=None,
                 draft_offsets=None,
                 retrieve_count=None
                 ):
        self.batch_size = batch_size
        self.token_count = token_count
        self.mode = mode  # 0:prefill 1:decode 10:mix
        self.samplings = samplings
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.pids = pids  # used for rope, record the start position id
        self.q_offsets = q_offsets
        self.k_offsets = k_offsets
        self.q_lengths = q_lengths
        self.k_lengths = k_lengths
        self.k_segs = k_segs
        self.max_q_length = max_q_length
        self.max_k_length = max_k_length
        self.max_seg = max_seg
        self.cache_indices = cache_indices  # used for cache update
        self.logit_indices = logit_indices  # used for cutoff logits
        self.embeddings = embeddings
        self.emb_idx_list = emb_idx_list
        self.reqs = reqs  # used for multi-node mode
        self.mask = mask
        self.qls = qls
        self.kls = kls
        self.spec = spec 
        self.draft_offsets = draft_offsets
        self.retrieve_count = retrieve_count

    @staticmethod
    def prefill_batching(reqs,
                         slots,
                         device=torch.device(0),
                         cache_size=None,
                         buffer=0,
                         min_rate=0.95,
                         allocate_rate=1.0,
                         fully_alloc_under=None,
                         embeddings=None):
        assert len(reqs) > 0
        if fully_alloc_under is None:
            fully_alloc_under = 2 ** 16
        if embeddings is not None and all([x is None for x in embeddings]):
            embeddings = None

        input_ids = []
        position_ids = []
        qls = []
        cache_offsets = []
        pids = []
        rs = []
        samplings = []
        emb_idx_list = []
        for ir, req in enumerate(reqs):

            if req.done == 0:  # no trunk or first chunk, should allocate slot
                if req.todo > 0:  # trunked
                    ql = req.todo
                else:
                    ql = req.input_length
                output_alloc_length = req.output_length if req.output_length <= fully_alloc_under else max(
                    int(allocate_rate * req.output_length), fully_alloc_under)
                alloc_length = ((req.input_length + int(
                    output_alloc_length) - 1 + buffer) // 16 + 1) * 16
                reserve = ((
                                       req.input_length + req.output_length - 1 + buffer) // 16 + 1) * 16
                cache_offset, slot_index = Batch.allocate(slots, 
                                                            alloc_length,
                                                          reserve=reserve,
                                                          cache_size=cache_size,
                                                          min_rate=min_rate)
                if cache_offset == -1:
                    for r in rs:
                        Batch.recycle(slots, r.segs)
                        r.todo = 0
                        r.done = 0
                    return Batch(batch_size=0)
                req.segs = [
                    (cache_offset, cache_offset + alloc_length, slot_index)]
                rs.append(
                    req)  # samples without allocating here should not be recycle
            else:  # done > 0, second or next trunks
                ql = req.todo
                cache_offset = req.segs[0][0]

            qls.append(ql)
            cache_offsets.append(cache_offset)
            if req.todo > 0:  # chunked or targeted
                if req.done < req.input_length:  # chunked
                    ids = req.input_ids[req.done:req.done + req.todo]
                    # position_ids.extend(range(req.done, req.done + req.todo))
                    pids.append(req.done)
                else:  # targeted
                    _, pos, ids = req.iterate_target()
                    ids = ids[:req.todo]
                    pids.append(req.input_length + pos)
                input_ids.extend(ids)
            else:
                assert req.done == 0
                input_ids.extend(req.input_ids)
                position_ids.extend(range(ql))
                pids.append(0)
            if req.target_ids or req.temperature or req.top_k or req.top_p or req.min_p:
                samplings.append(Sampling(target_ids=req.target_ids,
                                          temperature=req.temperature,
                                          top_k=req.top_k, 
                                          top_p=req.top_p,
                                          min_p=req.min_p))
            else:
                samplings.append(None)

            emb_idx = None
            start = sum(qls) - ql
            if embeddings is not None:
                if req.todo == 0:
                    ss = 0
                    se = req.emb_size
                    ds = start + req.emb_idx
                    de = start + req.emb_idx + req.emb_size
                    emb_idx = (ss, se, ds, de)
                elif req.done >= req.emb_idx + req.emb_size or req.done + req.todo <= req.emb_idx:
                    emb_idx = None
                else:
                    ss = max(req.done - req.emb_idx, 0)
                    se = min(req.done + req.todo - req.emb_idx, req.emb_size)
                    ds = start + max(req.emb_idx - req.done, 0)
                    de = start + min(req.emb_idx + req.emb_size - req.done, ql)
                    emb_idx = (ss, se, ds, de)
            emb_idx_list.append(emb_idx)

        if all([x is None for x in samplings]):
            samplings = None

        if all([x is None for x in emb_idx_list]):
            emb_idx_list = None

        accum = 0
        eo = 0
        cache_indices = []
        q_offsets = [0]
        k_offsets = []
        q_lengths = []
        k_lengths = []
        logit_indices = []
        for i, req in enumerate(reqs):
            targeted = req.done >= req.input_length
            ql = qls[i]
            cache_offset = cache_offsets[i]
            accum += ql
            q_offsets.append(accum)
            k_offsets.append(cache_offset)
            # a chunk that not contains the last \
            # token of a prompt does not need to calculate logits
            if targeted:
                pos = req.iterate_target()[1]
                cache_indices.extend(range(cache_offset + req.input_length + pos, cache_offset + req.input_length + pos + ql))
                logit_indices.extend(range(accum-ql, accum))
            else:
                # chunked 
                cache_indices.extend(range(cache_offset + req.done, cache_offset + req.done + ql))
                if req.done + ql == req.input_length:  
                    logit_indices.append(accum-1)
            used_k = pids[i] + ql
            q_lengths.append(used_k)
            k_lengths.append(used_k)

        k_offsets.append(cache_indices[-1])
        token_count = sum(qls)
        kls = q_lengths

        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.int32,
                                    device=device)
        pids = torch.tensor(pids, device=device, dtype=torch.int32)
        max_q_length = max(qls)
        max_k_length = max(k_lengths)
        q_offsets = torch.tensor(q_offsets, dtype=torch.int32, device=device)
        k_offsets = torch.tensor(k_offsets, dtype=torch.int32, device=device)
        k_lengths = torch.tensor(k_lengths, dtype=torch.int32, device=device)
        q_lengths = torch.tensor(q_lengths, dtype=torch.int32, device=device)
        cache_indices = torch.tensor(cache_indices,
                                              dtype=torch.int32, device=device)
        # may be empty
        logit_indices = torch.tensor(logit_indices, dtype=torch.int32,
                                     device=device)

        return Batch(batch_size=len(qls),
                     token_count=token_count,
                     mode=0,
                     input_ids=input_ids,
                     position_ids=position_ids,
                     pids=pids,
                     q_offsets=q_offsets,
                     k_offsets=k_offsets,
                     max_q_length=max_q_length,
                     max_k_length=max_k_length,
                     q_lengths=q_lengths,
                     k_lengths=k_lengths,
                     cache_indices=cache_indices,
                     logit_indices=logit_indices,
                     samplings=samplings,
                     embeddings=embeddings,
                     emb_idx_list=emb_idx_list,
                     reqs=reqs,
                     max_seg=1,
                     mask=None,
                     qls=qls,
                     kls=kls
                     )

    @staticmethod
    def decoding_batching(reqs, device=torch.device(0)):
        samplings = []

        bs = len(reqs)
        max_seg = max([len(x.segs) for x in reqs])
        qls = [1] * bs
        input_ids = [x.output_ids[-1] for x in reqs]

        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        position_ids = [x.input_length + len(x.output_ids) - 1 for x in reqs]
        max_q_length = 1
        max_k_length = max(position_ids) + 1
        if max_seg == 1:
            k_offsets = [x.segs[0][0] for x in reqs] + [reqs[-1].segs[0][0]]
        else:
            k_offsets = sum(
                [[y[0] for y in x.segs] + [0] * (max_seg - len(x.segs)) for x in
                 reqs], [])

        cache_indices = [k_offsets[i] + x for i, x in
                                  enumerate(position_ids)]

        q_lengths = [1] * bs
        k_segs = []
        if max_seg == 1:
            k_lengths = [x + 1 for x in position_ids]
        else:
            # accum lengths
            k_lengths = []
            for i in range(bs):
                segs = reqs[i].segs
                k_length = [0] + [x[1] - x[0] for x in segs[:-1]] + [
                    position_ids[i] + 1]
                k_length = k_length + [0] * (max_seg - len(segs))
                k_length = list(
                    itertools.accumulate(k_length, lambda x, y: x + y))
                k_lengths.append(k_length)
                k_segs.append(len(segs))

        kls = k_lengths
        position_ids = torch.tensor(position_ids, dtype=torch.int32,
                                    device=device)
        q_offsets = torch.arange(bs + 1, dtype=torch.int32, device=device)
        k_offsets = torch.tensor(k_offsets, dtype=torch.int32, device=device)
        q_lengths = torch.tensor(q_lengths, dtype=torch.int32, device=device)
        k_lengths = torch.tensor(k_lengths, dtype=torch.int32, device=device)
        k_segs = torch.tensor(k_segs, dtype=torch.int32, device=device)
        cache_indices = torch.tensor(cache_indices,
                                              dtype=torch.int32, device=device)

        for _, req in enumerate(reqs):
            if req.target_ids or req.temperature \
                 or req.top_k or req.top_p or req.min_p:
                samplings.append(Sampling(target_ids=req.target_ids,
                                          temperature=req.temperature,
                                          top_k=req.top_k, 
                                          top_p=req.top_p,
                                          min_p=req.min_p))
            else:
                samplings.append(None)

        if all([x is None for x in samplings]):
            samplings = None

        return Batch(batch_size=bs,
                     token_count=bs,
                     mode=1,
                     input_ids=input_ids,
                     position_ids=position_ids,
                     pids=position_ids,
                     q_offsets=q_offsets,
                     k_offsets=k_offsets,
                     max_q_length=max_q_length,
                     max_k_length=max_k_length,
                     q_lengths=q_lengths,
                     k_lengths=k_lengths,
                     cache_indices=cache_indices,
                     samplings=samplings,
                     reqs=reqs,
                     k_segs=k_segs,
                     max_seg=max_seg,
                     mask=None,
                     qls=qls,
                     kls=kls
                     )


    @staticmethod
    def lookahead_batching(reqs, 
                           spec, 
                           device=torch.device(0), 
                           retrieve_count=4):

        bs = len(reqs)

        retrieve_key_ids = [x.output_ids[-2:] if len(x.output_ids) >= 2 else [0,
                                                                              x.output_ids[
                                                                                  0]]
                            for x in reqs]
        input_ids, masks = retrieve_draft_table(retrieve_key_ids,
                                         spec.freq_table,
                                         spec.draft_table,
                                         spec.table_size,
                                         spec.branch_length,
                                         spec.branch_count,
                                         retrieve_count)
        input_ids = input_ids.view(-1)

        l = retrieve_count * spec.branch_length
        # max_seg = max([len(x.segs) for x in reqs])
        samplings = []
        qls = [l] * bs

        max_seg = max([len(x.segs) for x in reqs])
        position_ids = []
        pids = []
        draft_offsets = [0]  # used for rope
        cache_indices = []
        k_lengths = []
        k_offsets = []
        for i, req in enumerate(reqs):
            s = req.input_length + len(req.output_ids)
            n_seg = len(req.segs)
            offset = req.segs[-1][0]
            position_ids.extend(
                [s - 1 + j for j in range(l)])
            pids.extend([s] + [s+1]*(retrieve_count-1))
            draft_offsets.extend([i*l+(j+1)*spec.branch_length+(0 if j==retrieve_count-1 else 1) for j in range(retrieve_count)])
            if max_seg == 1:
                k_offsets.append(offset)
                k_lengths.append(position_ids[-1] + 1)
                cache_indices.extend([s - 1 + j + offset for j in
                                               range(
                                                   l)])
            else:
                k_offsets.extend([x[0] for x in req.segs]+[0]*(max_seg-n_seg))
                cache_indices.extend(
                    [s - 1 + j + offset[n_seg - 1] for j in
                     range(l)])
                k_length = [0] + [x[1] - x[0] for x in req.segs[:-1]] + [
                    position_ids[-1] + 1]
                k_length = k_length + [0] * (max_seg - len(req.segs))
                k_length = itertools.accumulate(k_length)
                k_lengths.extend(k_length)
        k_offsets.append(reqs[-1].segs[-1][0]+reqs[-1].input_length + len(reqs[-1].output_ids)+l-1)

        max_q_length = l
        max_k_length = max(position_ids) + 1

        position_ids = torch.tensor(position_ids, dtype=torch.int32,
                                    device=device)
        draft_offsets = torch.tensor(draft_offsets, dtype=torch.int32,
                                    device=device)
        pids = torch.tensor(pids, dtype=torch.int32,
                                    device=device)

        kls = k_lengths
        q_lengths = qls
        q_offsets = l * torch.arange(bs + 1, dtype=torch.int32, device=device)
        k_offsets = torch.tensor(k_offsets, dtype=torch.int32, device=device)
        q_lengths = torch.tensor(q_lengths, dtype=torch.int32, device=device)
        k_lengths = torch.tensor(k_lengths, dtype=torch.int32, device=device)
        cache_indices = torch.tensor(cache_indices,
                                              dtype=torch.int32, device=device)

        for _, req in enumerate(reqs):
            if req.target_ids or req.temperature or req.top_k or req.top_p or req.min_p:
                samplings.append(
                    Sampling(target_ids=req.target_ids,
                             temperature=req.temperature, top_k=req.top_k,
                             top_p=req.top_p,
                             min_p=req.min_p))
            else:
                samplings.append(None)

        if all([x is None for x in samplings]):
            samplings = None

        return Batch(batch_size=bs,
                     token_count=bs*l,
                     mode=2,
                     input_ids=input_ids,
                     position_ids=position_ids,
                     pids=pids,
                     q_offsets=q_offsets,
                     k_offsets=k_offsets,
                     max_q_length=max_q_length,
                     max_k_length=max_k_length,
                     q_lengths=q_lengths,
                     k_lengths=k_lengths,
                     cache_indices=cache_indices,
                     samplings=samplings,
                     reqs=reqs,
                     max_seg=max_seg,
                     mask=masks,
                     qls=qls,
                     kls=kls,
                     spec=spec, 
                     draft_offsets=draft_offsets, 
                     retrieve_count=retrieve_count
                     )


    @staticmethod
    def mix_batching(reqs, slots, device=torch.device(0), min_rate=0.95,
                     allocate_rate=1.0, fully_alloc_under=2 ** 16):
        assert len(reqs) > 0

        input_ids = []
        position_ids = []
        qls = []
        cache_offsets = []
        pids = []
        rs = []  # for recycle

        for req in reqs:
            if req.task_type == 1:
                input_ids.append(req.output_ids[-1])
                position_ids.append(req.input_length + len(req.output_ids) - 1)
                qls.append(1)
                cache_offsets.append(req.segs[0])
                pids.append(req.input_length + len(req.output_ids) - 1)
            else:

                if req.done == 0:  # complete sample or first chunk, should allocate slot
                    if req.todo > 0:  # trunked
                        ql = req.todo
                    else:  # complete sample
                        ql = req.input_length
                    output_alloc_length = req.output_length if req.output_length <= fully_alloc_under else allocate_rate * req.output_length
                    alloc_length = ((req.input_length + int(
                        output_alloc_length) - 1) // 16 + 1) * 16
                    cache_offset, slot_index = Batch.allocate(slots,
                                                              alloc_length,
                                                              min_rate=min_rate)
                    if cache_offset == -1:
                        for r in rs:
                            Batch.recycle(slots, r.segs)
                            r.todo = 0
                            r.done = 0
                            r.segs = None
                        return Batch(batch_size=0)
                    req.segs = (
                    cache_offset, cache_offset + alloc_length, slot_index)
                    rs.append(req)
                else:  # done > 0, i.e., second trunk
                    ql = req.todo
                    cache_offset = req.segs[0]

                qls.append(ql)
                cache_offsets.append(cache_offset)
                if req.todo > 0:  # chunked
                    ids = req.input_ids[req.done:req.done + req.todo]
                    input_ids.extend(ids)
                    position_ids.extend(range(req.done, req.done + req.todo))
                    pids.append(req.done)
                else:
                    assert req.done == 0
                    input_ids.extend(req.input_ids)
                    position_ids.extend(range(0, ql))
                    pids.append(0)

        accum = 0
        eo = 0
        cache_indices = []
        q_offsets = [0]
        k_offsets = []
        q_lengths = []
        k_lengths = []
        for i, req in enumerate(reqs):
            ql = qls[i]
            cache_offset = cache_offsets[i]
            accum += ql
            q_offsets.append(accum)
            so = cache_offset + pids[i]
            eo = so + ql
            cache_indices.extend(range(so, eo))
            k_offsets.append(cache_offset)
            k_lengths.append(pids[i] + ql)
            q_lengths.append(ql)
        k_offsets.append(eo)

        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.int32,
                                    device=device)
        pids = torch.tensor(pids, dtype=torch.int32, device=device)
        max_q_length = max(qls)
        max_k_length = max(k_lengths)
        kls = k_lengths

        q_offsets = torch.tensor(q_offsets, dtype=torch.int32, device=device)
        k_offsets = torch.tensor(k_offsets, dtype=torch.int32, device=device)
        q_lengths = torch.tensor(q_lengths, dtype=torch.int32, device=device)
        k_lengths = torch.tensor(k_lengths, dtype=torch.int32, device=device)
        cache_indices = torch.tensor(cache_indices,
                                              dtype=torch.int32, device=device)

        return Batch(batch_size=len(qls),
                     token_count=sum(qls),
                     mode=10,
                     input_ids=input_ids,
                     position_ids=position_ids,
                     pids=pids,
                     q_offsets=q_offsets,
                     k_offsets=k_offsets,
                     max_q_length=max_q_length,
                     max_k_length=max_k_length,
                     q_lengths=q_lengths,
                     k_lengths=k_lengths,
                     cache_indices=cache_indices,
                     reqs=reqs,
                     max_seg=1,
                     mask=None,
                     qls=qls,
                     kls=kls
                     )

    def to(self, device, non_blocking=False):
        if self.input_ids.device != device:
            self.input_ids = self.input_ids.to(device, non_blocking=non_blocking)
        self.input_ids = self.input_ids.to(device, non_blocking=non_blocking)
        self.position_ids = self.position_ids.to(device, non_blocking=non_blocking)
        self.pids = self.pids.to(device, non_blocking=non_blocking)
        self.q_offsets = self.q_offsets.to(device, non_blocking=non_blocking)
        self.k_offsets = self.k_offsets.to(device, non_blocking=non_blocking)
        self.q_lengths = self.q_lengths.to(device, non_blocking=non_blocking)
        self.k_lengths = self.k_lengths.to(device, non_blocking=non_blocking)
        self.cache_indices = self.cache_indices.to(device, non_blocking=non_blocking)
        if isinstance(self.logit_indices, torch.Tensor):
            self.logit_indices = self.logit_indices.to(device, non_blocking=non_blocking)
        if isinstance(self.mask, torch.Tensor) and self.mask.device != device:
            self.mask = self.mask.to(device, non_blocking=non_blocking)
        if isinstance(self.draft_offsets, torch.Tensor):
            self.draft_offsets = self.draft_offsets.to(device, non_blocking=non_blocking)

    def send(self, hidden_states, dst=1, group=None, light=False, log=False):
        dim = hidden_states.size(-1)
        dtype = 0 if hidden_states.dtype == torch.float16 else 1
        device = hidden_states.device
        comm_device = None
        objects = [
            [self.batch_size, self.token_count, self.mode, self.cache_slots,
             self.samplings, self.max_seqlen_q, self.max_seqlen_k, dtype, dim,
             self.reqs]]
        if log:
            print(f'start send objects:{objects}')
        dist.send_object_list(objects, dst=dst, group=group, device=comm_device)

        if light:
            return

        torch.cuda.synchronize(device=device)

        if log:
            print(f'start send input_ids:{self.input_ids}')
        dist.send(self.input_ids, dst, group=group)
        dist.send(self.position_ids, dst, group=group)
        dist.send(self.pids, dst, group=group)
        dist.send(self.q_offsets, dst, group=group)
        dist.send(self.k_offsets, dst, group=group)
        dist.send(self.k_lengths, dst, group=group)
        dist.send(self.cache_indices, dst, group=group)
        dist.send(self.logit_indices, dst, group=group)

        if log:
            print(f'start send hidden_states:{hidden_states[0, :3]}')
        dist.send(hidden_states, dst, group=group)
        torch.cuda.synchronize(device=device)

    def recv(self, src=0, group=None, light=False, log=False):
        objects = [None]
        device = 'cuda:0'
        comm_device = None
        dist.recv_object_list(objects, src=src, group=group, device=comm_device)
        objects = objects[0]
        self.batch_size = objects[0]
        self.token_count = objects[1]
        self.mode = objects[2]
        self.cache_slots = objects[3]
        self.samplings = objects[4]
        self.max_seqlen_q = objects[5]
        self.max_seqlen_k = objects[6]
        dtype = torch.float16 if objects[7] == 0 else torch.bfloat16
        dim = objects[8]
        self.reqs = objects[9]
        if log:
            print(f'finish recv objects:{objects}')

        if light:
            return

        token_count = self.token_count
        batch_size = self.batch_size

        self.input_ids = torch.empty([token_count], device=device,
                                     dtype=torch.int32)
        self.position_ids = torch.empty([token_count], device=device,
                                        dtype=torch.int32)
        self.pids = torch.empty([batch_size], device=device, dtype=torch.int32)
        self.q_offsets = torch.empty([batch_size + 1], device=device,
                                     dtype=torch.int32)
        self.k_offsets = torch.empty([batch_size + 1], device=device,
                                     dtype=torch.int32)
        self.k_lengths = torch.empty([batch_size], device=device,
                                     dtype=torch.int32)
        self.cache_indices = torch.empty([token_count], device=device,
                                         dtype=torch.int32)
        self.logit_indices = torch.empty([batch_size], device=device,
                                         dtype=torch.int32)
        hidden_states = torch.empty([token_count, dim], device=device,
                                    dtype=dtype)
        torch.cuda.synchronize(device=device)

        dist.recv(self.input_ids, src=src, group=group)
        if log:
            print(f'finish recv input_ids:{self.input_ids}')

        dist.recv(self.position_ids, src=src, group=group)
        dist.recv(self.pids, src=src, group=group)
        dist.recv(self.q_offsets, src=src, group=group)
        dist.recv(self.k_offsets, src=src, group=group)
        dist.recv(self.k_lengths, src=src, group=group)
        dist.recv(self.cache_indices, src=src, group=group)
        dist.recv(self.logit_indices, src=src, group=group)

        # meta_tensor = torch.empty([token_count*3+batch_size*4+2], device=device, dtype=torch.int32)

        # hidden_states = torch.empty([token_count, dim], device=device, dtype=dtype)
        # torch.cuda.synchronize(device=device)
        # dist.recv(meta_tensor, src=src, group=group)
        # self.input_ids = meta_tensor[:token_count]
        # self.position_ids = meta_tensor[token_count:2*token_count]
        # self.pids = meta_tensor[2*token_count:2*token_count+batch_size]
        # self.cu_seqlens_q = meta_tensor[2*token_count+batch_size:2*token_count+2*batch_size+1]
        # self.cu_seqlens_k = meta_tensor[2*token_count+2*batch_size+1:2*token_count+3*batch_size+2]
        # self.seqused_k = meta_tensor[2*token_count+3*batch_size+2:2*token_count+4*batch_size+2]
        # self.cache_indices = meta_tensor[2*token_count+4*batch_size+2:3*token_count+4*batch_size+2]

        dist.recv(hidden_states, src=src, group=group)
        if log:
            print(f'finish recv hidden_states:{hidden_states[0, :4]}')

        torch.cuda.synchronize(device=device)
        return hidden_states

    @staticmethod
    def allocate(slots, length, reserve=None, cache_size=None, min_rate=0.95):
        max_end_idx = cache_size - reserve if reserve is not None else 2 ** 30
        with slots.get_lock():
            rates = np.zeros(len(slots), dtype=np.float32)
            max_idx = -1
            max_rate = 0.0
            for j, slot in enumerate(slots):
                s, e, state = slot.s, slot.e, slot.state
                if state == 1 and length <= e - s and s <= max_end_idx:
                    rate = length / (e - s)
                    if rate >= min_rate:
                        max_idx = j
                        max_rate = rate
                        break
                    rates[j] = rate
                elif s == 0 and e == 0 and state == 0:
                    break

            if max_idx == -1:
                max_idx = np.argmax(rates)
                max_rate = rates[max_idx]

            if max_rate > 0.0:
                slot = slots[max_idx]
                s, e, state = slot.s, slot.e, slot.state
                # print(f"pid:{os.getpid()} allocate:[{s},{s + length}] segment:[{s+length},{e}]")
                if s + length == e:  # accurate matched
                    slot.state = 2  # occupied
                else:
                    slot.e = s + length
                    slot.state = 2  # occupied
                    for iter_slot in slots:
                        if iter_slot.state == 0:
                            iter_slot.s = s + length
                            iter_slot.e = e
                            iter_slot.state = 1  # available
                            break
                return s, max_idx
            else:
                return -1, -1

    @staticmethod
    def recycle(slots, segs):  # be carefull with prefix cache
        with slots.get_lock():
            for ts, te, idx in segs:
                if slots[idx].share:
                    continue
                org_state = slots[idx].state
                assert org_state in (2, 4)
                if org_state == 4:  # keep reserved and do not merge
                    slots[idx].state = 3  # available and reversed
                    return
                slots[idx].state = 0
                cs = None
                for slot in slots:
                    if slot.state == 1 and slot.e == ts:
                        cs = slot.s
                        slot.state = 0
                        break
                    elif slot.s == 0 and slot.e == 0 and slot.state == 0:
                        break

                ce = None
                for slot in slots:
                    if slot.state == 1 and slot.s == te:
                        ce = slot.e
                        slot.state = 0
                        break
                    elif slot.s == 0 and slot.e == 0 and slot.state == 0:
                        break

                s = cs if cs is not None else ts
                e = ce if ce is not None else te

                hit = False
                for slot in slots:
                    if slot.state == 0:
                        slot.s = s
                        slot.e = e
                        slot.state = 1
                        hit = True
                        break

                if hit is False:
                    raise ValueError("Recycle error! No slot available!")

    @staticmethod
    def extend_slot(slots, old_segs, length, contiguous=False):
        # contiguous: use single-seg if True else use multi-seg
        if contiguous:
            """
            state=0: undefined
            state=1: available
            state=2: occupied
            state=3: available and reserved
            state=4: occupied and reserved
            a cache memory must belongs one of states [1 2 3 4]
            """
            assert len(old_segs) == 1
            with slots.get_lock():
                old_seg = old_segs[0]
                ts, te, idx = old_seg
                for slot in slots:
                    if slot.s == te and slot.state in (1, 3):
                        if slot.e >= te + length:
                            if slot.e == te + length:
                                slot.state = 0
                            else:
                                # shrink following slot
                                slot.s = te + length
                                if slot.state == 3:
                                    slot.state = 1  # become available if state==3
                            slots[idx].e = te + length  # extend previous slot
                            return [(ts, te + length, idx)]  # fully allocated
                        else:
                            slot.state = 0
                            slots[idx].e = slot.e
                            return [(ts, slot.e, idx)]  # partly allocated
                    elif slot.s == te and slot.state in (2, 4):
                        slot.state = 4  # labeled as occupied and reserved
                        return None
                    elif slot.s == 0 and slot.e == 0 and slot.state == 0:
                        return None

        else:

            """
            state=0: undefined
            state=1: available
            state=2: occupied
            """

            with slots.get_lock():
                old_seg = old_segs[-1]
                ts, te, idx = old_seg
                success = True
                # try to extend
                for slot in slots:
                    if slot.s == te and slot.state in (1,):
                        if slot.e >= te + length:
                            if slot.e == te + length:
                                slot.state = 0
                            else:
                                slot.s = te + length
                                slot.state = 1
                            slots[idx].e = te + length
                            new_segs = old_segs[:-1] + [(ts, te + length, idx)]
                            return new_segs  # fully allocated
                        else:
                            slot.state = 0
                            slots[idx].e = slot.e
                            new_segs = old_segs[:-1] + [(ts, slot.e, idx)]
                            return new_segs  # partly allocated
                    elif slot.s == te and slot.state == 2:
                        success = False
                        break
                    elif slot.s == 0 and slot.e == 0 and slot.state == 0:
                        success = False
                        break

                if not success:  # use an uncontiguous seg
                    for idx, slot in enumerate(slots):
                        if slot.state == 1:
                            if slot.e - slot.s >= length:
                                slot.state = 2
                                if slot.e - slot.s > length:
                                    for iter_slot in slots:
                                        if iter_slot.state == 0:
                                            iter_slot.s = slot.s + length
                                            iter_slot.e = slot.e
                                            iter_slot.state = 1  # available
                                            break
                                return old_segs + [
                                    (slot.s, slot.s + length, idx)]
                            else:
                                slot.state = 2
                                return old_segs + [(slot.s, slot.e, idx)]

    @staticmethod
    def slot_check(slots):
        counts = [0, 0, 0, 0, 0]
        sizes = [0, 0, 0, 0, 0]
        for slot in slots:
            counts[slot.state] += 1
            sizes[slot.state] += slot.e - slot.s
        return {"counts": counts, "sizes": sizes}
