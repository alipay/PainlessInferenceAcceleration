# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import enum
import math
import copy
from ctypes import Structure, c_int
import itertools

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
                 logit_counts=None,
                 reqs=None,
                 qls=None,
                 kls=None,
                 spec=None,
                 draft_offsets=None,
                 retrieve_count=None,
                 embeddings=None,
                 emb_idx_list=None
                 ):
        """
        batch meta info for forward
        :param batch_size: req count
        :param token_count: token count
        :param mode: forwar mode, 0:prefill 1:decode 2:spec 10:mix
        :param samplings: sampling and target params
        :param input_ids: input_id tensor
        :param position_ids: used for rope, only record the start position id of each req
            example: a chunked prefill query is start from 1024 to 2048
            position_ids = [1024]
        :param q_offsets: the first index for query tokens of each req in a batch
            shape of [bs+1], the last one is meaningless, only to be compatible for flash-attention
            decode example: [0,1,2,...,bs]
            prefill example: [0, q0, q0+q1, q0+q1+q2,...]
        :param k_offsets: the first key/value indices of each req
            SINGLE SEG: the kvcaches of reqs are all single-segment
                shape of [bs+1], the last one is meaningless
                example: [4096,1024,2048,...,dummy]
            MULTIPLE SEG: at least one of the kvcaches of reqs are multi-segment
                shape of [bs, max_seg]
                example:  
                    first req has 2 segs with indices [4096,1024] and second req has 1 seg with indices [3073]
                    k_offsets = [[4096,1024],[3072,0]]
        :param max_q_length: the max query length of queries in a batch
            it is used for computing sm
        :param max_k_length: the max k length, used in flash-attn
        :param k_segs: the seg counts of each req, shape of [bs]
        :param max_seg: max seg of reqs
        :param q_lengths: query length of each req, shape of [bs]
        :param k_lengths: kvcache size of each req
            SINGLE SEG: the kvcaches of reqs are all single-segment
                shape of [bs]
                example: [1024,1024,1024,...]
            MULTIPLE SEG: at least one of the kvcaches of reqs are multi-segment
                shape of [bs, max_seg+1], the last one is total length
                example:
                    first req's segments: [1024, 512]
                    second req's segments: [512]
                    k_lengths = [[1024,512,1536],[512,0,512]]
        :param mask: attention mask
        :param cache_indices: kv cache indices, used for cache updating,
            it maps kvcache of current req to cache memory
        :param logit_indices: indices that need to calculate logit,
            it is used for cutoff hidden_states of the last layer to avoid redundant calculation
        :param logit_counts: count of calculated logits of each req,
            it is used to map logits and output ids of a req in sampling process
        :param reqs: reqs,  `output_ids` of reqs will be updated with next token ids
        :param qls: python list of query lengths
        :param kls: python list of kvcache sizes, it is list of list if multi-segment
        :param spec: spec instance
        :param draft_offsets: act as position ids for each draft
        :param retrieve_count: lookahead retrieve draft count
        :param embeddings: embeddings of multi-modal models
        :param emb_idx_list: embedding positions of multi-modal models
        """
        self.batch_size = batch_size
        self.token_count = token_count
        self.mode = mode  
        self.samplings = samplings
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.q_offsets = q_offsets
        self.k_offsets = k_offsets
        self.q_lengths = q_lengths
        self.k_lengths = k_lengths
        self.k_segs = k_segs
        self.max_q_length = max_q_length
        self.max_k_length = max_k_length
        self.max_seg = max_seg
        self.cache_indices = cache_indices
        self.logit_indices = logit_indices
        self.logit_counts = logit_counts
        self.reqs = reqs
        self.mask = mask
        self.qls = qls
        self.kls = kls
        self.spec = spec 
        self.draft_offsets = draft_offsets
        self.retrieve_count = retrieve_count
        self.embeddings = embeddings
        self.emb_idx_list = emb_idx_list

    @staticmethod
    def prefill_batching(reqs,
                         slots,
                         fixed_slots,
                         device=torch.device(0),
                         cache_size=None,
                         buffer_size=0,
                         min_rate=0.95,
                         fully_alloc_under=None,
                         embeddings=None):
        assert len(reqs) > 0
        if fully_alloc_under is None:
            fully_alloc_under = 2 ** 16
        if embeddings is not None and all([x is None for x in embeddings]):
            embeddings = None

        input_ids = []
        position_ids = []
        cache_indices = []
        q_offsets = [0]
        k_offsets = []
        q_lengths = []
        k_lengths = []
        logit_indices = []
        logit_counts = []
        allocated = [] # allocated samples should be recycle if failed in a batch
        samplings = []
        emb_idx_list = []
        for i, req in enumerate(reqs):

            if req.done == 0:  # new req or first chunk, should allocate slot
                if req.todo > 0:  # trunked
                    ql = req.todo
                else:
                    ql = req.input_length
                output_alloc_length = min(req.output_length, fully_alloc_under)
                total_alloc_length = Batch.cdiv(req.input_length + output_alloc_length + buffer_size, 16)
                if output_alloc_length < req.output_length:
                    reserve = Batch.cdiv(req.output_length - output_alloc_length, 16)
                else:
                    reserve = buffer_size
                cache_offset, slot_index = Batch.allocate(slots, 
                                                          fixed_slots,
                                                          total_alloc_length,
                                                          reserve=reserve,
                                                          cache_size=cache_size,
                                                          min_rate=min_rate)
                if cache_offset == -1:  # failed
                    for r in allocated:
                        Batch.recycle(slots, r.segs)
                        r.todo = 0
                        r.done = 0
                    return Batch(batch_size=0)
                req.segs = [
                    (cache_offset, cache_offset + total_alloc_length, slot_index)]
                allocated.append(req)  
            else:  # done > 0, second trunk
                ql = req.todo
                cache_offset = req.segs[0][0]

            q_lengths.append(ql)
            sum_ql = sum(q_lengths)
            q_offsets.append(sum_ql)  # next q_offset
            k_offsets.append(cache_offset)

            targeted = req.done >= req.input_length
            # a chunk that not contains the last \
            # token of a prompt does not need to calculate logits
            if targeted:
                pos = req.iterate_target()[1]
                cache_indices.extend(range(cache_offset + req.input_length + pos, cache_offset + req.input_length + pos + ql))
                logit_indices.extend(range(sum_ql-ql, sum_ql))
                logit_counts.append(ql)
            else:
                # chunked 
                cache_indices.extend(range(cache_offset + req.done, cache_offset + req.done + ql))
                if req.done + ql == req.input_length:  
                    logit_indices.append(sum_ql-1)
                    logit_counts.append(1)
                else:
                    logit_counts.append(0)

            if req.todo > 0:  # chunked or targeted
                if req.done < req.input_length:  # chunked
                    ids = req.input_ids[req.done:req.done + req.todo]
                    position_id = req.done
                else:  # targeted
                    _, pos, ids = req.iterate_target()
                    ids = ids[:req.todo]
                    position_ids = req.input_length + pos
                input_ids.extend(ids)
            else:  # complelte req
                assert req.done == 0
                input_ids.extend(req.input_ids)
                position_id = 0
            position_ids.append(position_id)
            k_lengths.append(position_id + ql)

            if req.target_ids or req.temperature or req.top_k or req.top_p or req.min_p:
                samplings.append(Sampling(target_ids=req.target_ids,
                                          temperature=req.temperature,
                                          top_k=req.top_k, 
                                          top_p=req.top_p,
                                          min_p=req.min_p))
            else:
                samplings.append(None)

            emb_idx = None
            start = sum(q_lengths) - ql
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

        k_offsets.append(cache_indices[-1]+q_lengths[-1])
        token_count = sum(q_lengths)
        kls = k_lengths
        qls = q_lengths

        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        position_ids = torch.tensor(position_ids, device=device, dtype=torch.int32)
        max_q_length = max(q_lengths)
        max_k_length = max(k_lengths)
        q_offsets = torch.tensor(q_offsets, dtype=torch.int32, device=device)
        k_offsets = torch.tensor(k_offsets, dtype=torch.int32, device=device)
        q_lengths = torch.tensor(q_lengths, dtype=torch.int32, device=device)
        k_lengths = torch.tensor(k_lengths, dtype=torch.int32, device=device)
        cache_indices = torch.tensor(cache_indices,
                                              dtype=torch.int32, device=device)
        # NOTE: may be empty
        logit_indices = torch.tensor(logit_indices, dtype=torch.int32,
                                     device=device)

        return Batch(batch_size=len(q_lengths),
                     token_count=token_count,
                     mode=0,
                     input_ids=input_ids,
                     position_ids=position_ids,
                     q_offsets=q_offsets,
                     k_offsets=k_offsets,
                     max_q_length=max_q_length,
                     max_k_length=max_k_length,
                     q_lengths=q_lengths,
                     k_lengths=k_lengths,
                     cache_indices=cache_indices,
                     logit_indices=logit_indices,
                     logit_counts=logit_counts,
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
        input_ids = [x.output_ids[-1] for x in reqs]

        position_ids = []
        cache_indices = []
        k_offsets = []
        k_lengths = []
        k_segs = []
        for i in range(bs):
            req = reqs[i]
            segs = req.segs
            n_seg = len(segs)
            k_segs.append(n_seg)
            kl = req.input_length + len(req.output_ids)
            position_id = kl - 1
            position_ids.append(position_id)
            if max_seg == 1:
                k_offset = segs[0][0]
                cache_index = k_offset + position_id
                k_length = kl
            else:
                k_offset = [x[0] for x in segs] + [0] * (max_seg - n_seg)
                pre_length = sum([x[1]-x[0] for x in segs[:-1]])
                cur_length = kl - pre_length 
                cache_index = segs[-1][0] + cur_length - 1
                k_length = [x[1] - x[0] for x in segs[:-1]] + [cur_length] + [0] * (max_seg - n_seg) + [kl]

            k_offsets.append(k_offset)
            if i == bs - 1 and max_seg == 1:  # for flash-attn
                k_offsets.append(k_offset + kl)
            cache_indices.append(cache_index)
            k_lengths.append(k_length)

        max_q_length = 1
        max_k_length = max(position_ids) + 1 

        qls = [1] * bs
        kls = k_lengths
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.int32,
                                    device=device)
        q_offsets = torch.arange(bs + 1, dtype=torch.int32, device=device)
        k_offsets = torch.tensor(k_offsets, dtype=torch.int32, device=device)
        q_lengths = torch.tensor(qls, dtype=torch.int32, device=device)
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
        draft_offsets = [0]  # used for rope
        cache_indices = []
        k_lengths = []
        k_offsets = []
        for i, req in enumerate(reqs):
            s = req.input_length + len(req.output_ids)
            n_seg = len(req.segs)
            offset = req.segs[-1][0]
            position_ids.extend([s] + [s+1]*(retrieve_count-1))
            draft_offsets.extend([i*l+(j+1)*spec.branch_length+(0 if j==retrieve_count-1 else 1) for j in range(retrieve_count)])
            if max_seg == 1:
                k_offsets.append(offset)
                k_lengths.append(s + l - 1)
                cache_indices.extend([s - 1 + j + offset for j in
                                               range(
                                                   l)])
            else:
                k_offsets.extend([x[0] for x in req.segs]+[0]*(max_seg-n_seg))
                cache_indices.extend(
                    [s - 1 + j + offset[n_seg - 1] for j in
                     range(l)])
                k_length = [0] + [x[1] - x[0] for x in req.segs[:-1]] + [
                    s + l - 1]
                k_length = k_length + [0] * (max_seg - len(req.segs))
                k_length = itertools.accumulate(k_length)
                k_lengths.extend(k_length)
        k_offsets.append(reqs[-1].segs[-1][0]+reqs[-1].input_length + len(reqs[-1].output_ids)+l-1)

        max_q_length = l
        max_k_length = max(k_lengths)  # NOTE: CHANGED AND NOT VERIFY 

        draft_offsets = torch.tensor(draft_offsets, dtype=torch.int32,
                                    device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.int32,
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
        rs = []  # for recycle

        for req in reqs:
            if req.task_type == 1:
                input_ids.append(req.output_ids[-1])
                qls.append(1)
                cache_offsets.append(req.segs[0])
                position_ids.append(req.input_length + len(req.output_ids) - 1)
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
                    position_ids.append(req.done)
                else:
                    assert req.done == 0
                    input_ids.extend(req.input_ids)
                    position_ids.append(0)

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
            so = cache_offset + position_ids[i]
            eo = so + ql
            cache_indices.extend(range(so, eo))
            k_offsets.append(cache_offset)
            k_lengths.append(position_ids[i] + ql)
            q_lengths.append(ql)
        k_offsets.append(eo)

        input_ids = torch.tensor(input_ids, dtype=torch.int32, device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.int32, device=device)
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
        self.input_ids = self.input_ids.to(device, non_blocking=non_blocking)
        self.position_ids = self.position_ids.to(device, non_blocking=non_blocking)
        self.q_offsets = self.q_offsets.to(device, non_blocking=non_blocking)
        self.k_offsets = self.k_offsets.to(device, non_blocking=non_blocking)
        self.q_lengths = self.q_lengths.to(device, non_blocking=non_blocking)
        self.k_lengths = self.k_lengths.to(device, non_blocking=non_blocking)
        self.cache_indices = self.cache_indices.to(device, non_blocking=non_blocking)
        if isinstance(self.k_segs, torch.Tensor):
            self.k_segs = self.k_segs.to(device, non_blocking=non_blocking)    
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
            [self.batch_size, self.token_count, self.mode, self.cache_indices,
             self.samplings, self.max_q_length, self.max_k_length, dtype, dim,
             self.reqs, self.qls, self.kls, self.max_seg, self.k_segs, self.logit_indices]]
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
        dist.send(self.q_offsets, dst, group=group)
        dist.send(self.k_offsets, dst, group=group)
        dist.send(self.q_lengths, dst, group=group)
        dist.send(self.k_lengths, dst, group=group)
        dist.send(self.cache_indices, dst, group=group)

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
        self.cache_indices = objects[3]
        self.samplings = objects[4]
        self.max_q_length = objects[5]
        self.max_k_length = objects[6]
        dtype = torch.float16 if objects[7] == 0 else torch.bfloat16
        dim = objects[8]
        self.reqs = objects[9]
        self.qls = objects[10]
        self.kls = objects[11]
        self.max_seg = objects[12]
        self.k_segs = objects[13]
        self.logit_indices = objects[14]
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
        self.q_offsets = torch.empty([batch_size + 1], device=device,
                                     dtype=torch.int32)
        self.k_offsets = torch.empty([batch_size + 1], device=device,
                                     dtype=torch.int32)
        self.q_lengths = torch.empty([batch_size], device=device,
                                     dtype=torch.int32)
        self.k_lengths = torch.empty([batch_size], device=device,
                                     dtype=torch.int32)
        self.cache_indices = torch.empty([token_count], device=device,
                                         dtype=torch.int32)
        hidden_states = torch.empty([token_count, dim], device=device,
                                    dtype=dtype)
        torch.cuda.synchronize(device=device)

        dist.recv(self.input_ids, src=src, group=group)
        if log:
            print(f'finish recv input_ids:{self.input_ids}')

        dist.recv(self.position_ids, src=src, group=group)
        dist.recv(self.q_offsets, src=src, group=group)
        dist.recv(self.k_offsets, src=src, group=group)
        dist.recv(self.q_lengths, src=src, group=group)
        dist.recv(self.k_lengths, src=src, group=group)
        dist.recv(self.cache_indices, src=src, group=group)

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
    def allocate(slots, fixed_slots, length, reserve=None, cache_size=None, min_rate=0.95):
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
            # with fixed_slots.get_lock():
            #     for j, fixed_slot in enumerate(fixed_slots):
            #         if fix_size_slot_index is not None:
            #             if fixed_slot.state != 1:
            #                 fix_size_slot_index = fixed_slot.fix_size_slot_index
            #             fixed_slot.state = 2

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
        old_segs = copy.deepcopy(old_segs)
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

    @staticmethod 
    def cdiv(x, b):
        return ((x-1)//b+1)*b