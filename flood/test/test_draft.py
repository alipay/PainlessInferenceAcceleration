# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from flood.ops.draft import *


def benchmark_func(fn, *args, n_warmup=100, n_repeat=1000, ref_time=None,
                   desc='', **kwargs):
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

    print(f'{func_name} {desc} time:{average_event_time:.1f} us')
    return average_event_time


if __name__ == '__main__':
    size = 2 ** 24
    length = 8
    count = 8
    freq_table = torch.zeros((size,), dtype=torch.float32, device='cuda:0')
    draft_table = torch.zeros((size, length), dtype=torch.int32,
                              device='cuda:0')
    tokens = list(range(10000))
    update_draft_table(tokens, freq_table, draft_table, table_size=size,
                       branch_length=length, branch_count=count, vocab=128256, eos=0)


    # print(freq_table[freq_table>0])
    # print(draft_table[draft_table[:,1]>0])

    # benchmark_func(update_draft_table, tokens, freq_table, draft_table, size=size, length=length, count=count)

    batch_size = 32
    retrieve_count = 4
    retrieve_length = 4
    tokens = [[i * 2, i * 2 + 1] for i in range(batch_size)]
    output_tokens, output_masks = retrieve_draft_table(tokens, 
                                                       freq_table,
                                                       draft_table, 
                                                       table_size=size,
                                                       vocab=128256,
                                                       branch_length=length,
                                                       branch_count=count,
                                                       retrieve_count=retrieve_count)
    print(output_tokens)
    print(output_masks)
    benchmark_func(retrieve_draft_table, tokens, freq_table, draft_table,
                   size=size, length=length, count=count,
                   retrieve_count=retrieve_count,
                   retrieve_length=retrieve_length, vocab=100000, eos=1000)
