# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import asyncio
import json
import random
import time
from typing import List

import aiohttp

from flood.common.llm import log
from flood.utils.reader import Reader

random.seed(7)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def async_request(req):
    data = {"prompt": req.input_text, "max_length": req.output_length}
    ttft = 0.0
    ts = time.time()
    token_count = 0
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url="http://localhost:30010/generate",
                                json=data, chunked=True,
                                read_bufsize=1) as response:
            async for chunk_bytes in response.content.iter_chunked(256):
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk_bytes = chunk_bytes.decode("utf-8")
                try:
                    data = json.loads(chunk_bytes)
                except:
                    print(f"error with chunk:{chunk_bytes}")
                    token_count += 11
                    continue
                if token_count == 0:
                    ttft = time.time() - ts
                token_count += data["token_count"]
        latency = time.time() - ts
    return ttft, latency, token_count


async def get_request(reqs, request_rate: float = 1.0, ):
    for request in reqs:
        yield request

        if request_rate == float("inf"):
            continue

        await asyncio.sleep(1.0 / request_rate)


async def benchmark(reqs, request_rate: float = 1.0):
    n_input = len(reqs)
    ts = time.time()
    tasks: List[asyncio.Task] = []
    async for request in get_request(reqs, request_rate=request_rate):
        tasks.append(asyncio.create_task(async_request(request)))
    outputs = await asyncio.gather(*tasks)
    bench_time = time.time() - ts
    n_output = len(outputs)
    assert n_input == n_output
    ttfts = 0
    latencies = 0
    token_counts = 0
    for ttft, latency, token_count in outputs:
        ttfts += ttft
        latencies += latency
        token_counts += token_count
    input_length = sum([req.input_length for req in reqs]) / n_output
    output_length = token_counts / n_output
    line = f'sample:{n_output} length:{input_length:.0f}/{output_length:.0f} qps:{request_rate:.2f} duration:{bench_time:.1f}s ttft:{ttfts / n_output:.3f}s latency:{latencies / n_output:.3f}s throughput:{token_counts / bench_time:.2f}token/s'
    with open('server.log', 'a+') as logger:
        log(logger, line)


if __name__ == '__main__':
    # model_path = '/mnt/nas_acr89/jingyue/med-fp8'  # llama(fp8 dynamic)
    # model_path = '/mnt/prev_nas/nanxiao/llama3'  # llama(empty)

    ils = [100, 500, 1000, 2000, 4000, 16384, 16384, 32768, 65536]
    ols = [300, 300, 300, 500, 500, 1, 1, 1, 1]
    cs = [10000, 5000, 5000, 5000, 2500, 1, 5, 5, 5]
    # request_rates = [16, 7.7, 4.3, 1.9, 0.9]
    request_rates = [16.5, 7.8, 4.4, 2.0, 0.95, 0.2, 0.2, 0.2, 0.2]
    for i in range(5, 9):
        data_path = 'dummy'
        reqs = Reader.read_dummy_dataset(max_count=cs[i], input_length=ils[i],
                                         output_length=ols[i], flunc=0.0)

        # data_path = '/mntnlp/nanxiao/dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'
        # reqs = Reader.read_sharegpt_dataset(data_path, model_path, max_count=2000)

        # sort_by = 'random'
        # reqs = Reader.sort_by(reqs, key=sort_by)

        benchmark_result = asyncio.run(
            benchmark(reqs, request_rate=request_rates[i]))
