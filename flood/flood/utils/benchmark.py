# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import time
import random

import torch
import numpy as np


def benchmark_func(fn, *args, n_warmup=100, n_repeat=1000, ref_time=None,
                   ref_flops=None, desc='', **kwargs):
    func_name = fn.__name__

    for i in range(n_warmup):
        fn(*args, **kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in
                    range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in 
                    range(n_repeat)]

    for i in range(n_repeat):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1, n_repeat // 100)
    times = sum(times[clip:-clip])

    average_event_time = times * 1000 / (n_repeat - 2 * clip)

    line = f'{func_name} {desc} time:{average_event_time:.1f} us'

    if ref_flops is not None:
        perf = ref_flops / 1e12 / (
                average_event_time / 1e6)
        line += f' perf:{perf:.2f} TFLOPS'

    if ref_time is not None:
        speedup = ref_time / average_event_time 
        line += f' speedup:{speedup:.2f}'
    
    print(line)
    return average_event_time


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True