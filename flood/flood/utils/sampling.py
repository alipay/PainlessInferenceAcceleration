# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


class Sampling:
    def __init__(self, temperature=1.0, top_k=None, top_p=None, target_ids=None,
                 min_p=None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.target_ids = target_ids
        self.min_p = min_p
