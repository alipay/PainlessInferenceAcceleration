# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


class Req:
    def __init__(self, rid, input_ids=None, input_length=-1, output_length=200,
                 done=0, todo=0, task_type=0, stream=False, emb_idx=0,
                 emb_size=0, segs=None, output_index=0, target_ids=None,
                 temperature=None, top_k=None, top_p=None, min_p=None):
        # used in queue, remove unused fields to reduce pickle and unpickle time
        self.rid = rid
        self.input_ids = input_ids
        self.input_length = input_length
        self.output_length = output_length
        self.done = done
        self.todo = todo
        self.task_type = task_type
        self.stream = stream
        self.emb_idx = emb_idx
        self.emb_size = emb_size
        self.segs = segs
        self.output_index = output_index
        self.target_ids = target_ids
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p

        self.output_ids = []

    def __repr__(self):
        return f'Req(rid={self.rid}, input_ids={self.input_ids}, output_ids={self.output_ids}, input_length={self.input_length}, output_length={self.output_length})'

    def size_of_segs(self):
        return sum([x[1] - x[0] for x in self.segs])

    def iterate_target(self):
        index = 0
        for i, target in enumerate(self.target_ids):
            if index + self.input_length <= self.done < index + self.input_length + len(target):
                offset = self.done - index - self.input_length
                return i, offset, target[offset:]
            index += len(target)
        return None, None, None

class Request:
    def __init__(self, rid, input_text=None, input_ids=None, output_text=None,
                 output_ids=None, input_length=-1, output_length=200, emb_idx=0,
                 emb_size=0, content=None, output_index=0, target_tokens=None,
                 target_ids=None, temperature=None, top_k=None, top_p=None,
                 min_p=None):
        # used in samples
        self.rid = rid
        self.input_text = input_text
        self.input_ids = input_ids
        self.output_text = output_text
        self.output_ids = output_ids
        self.input_length = input_length
        self.output_length = output_length
        self.emb_idx = emb_idx
        self.emb_size = emb_size
        self.content = content
        self.output_index = output_index
        self.target_ids = target_ids
        self.target_tokens = target_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
