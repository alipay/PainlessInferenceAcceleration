# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from flood.ops.sample import sample_from_logit


def benchmark_func(fn, *args, n_warmup=100, n_repeat=1000, ref_time=None,
                   ref_flops=None, desc='', **kwargs):
    func_name = fn.__name__

    for i in range(n_warmup):
        fn(*args, **kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in
                    range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

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

    speedup = ref_time / average_event_time if ref_time is not None else 1.0
    flops = ref_flops / 1e12 / (
                average_event_time / 1e6) if ref_flops is not None else 0.0
    print(
        f'{func_name} {desc} time:{average_event_time:.1f} us flops:{flops:.2f} TFLOPS speedup:{speedup:.2f}')
    return average_event_time


if __name__ == '__main__':
    device = 'cuda:0'
    dtype = torch.float32
    bs = 256
    logits = torch.randn((bs, 126464), dtype=dtype, device=device)
    temperature = [1 + random.random() for x in range(bs)]
    top_k = [1 + int(3 * random.random()) for x in range(bs)]
    top_p = [1 - 0.5 * random.random() for x in range(bs)]
    max_top_k = max(top_k)
    temperature = torch.tensor(temperature, dtype=dtype, device=device)
    top_k = torch.tensor(top_k, dtype=torch.int32, device=device)
    top_p = torch.tensor(top_p, dtype=dtype, device=device)

    outputs = sample_from_logit(logits, temperature, top_k, top_p, max_top_k)
    # print(outputs)

    benchmark_func(torch.argmax, logits)
    benchmark_func(torch.topk, logits, max_top_k, dim=-1, largest=True,
                   sorted=True)
    benchmark_func(sample_from_logit, logits, temperature, top_k, top_p,
                   max_top_k)
