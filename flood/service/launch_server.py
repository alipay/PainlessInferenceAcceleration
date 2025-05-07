# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import json

import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from flood.facade.llm import LLM
from flood.utils.request import Request as FloodRequest

app = FastAPI()


@app.post("/generate")
async def openai_v1_completions(request: Request):
    req = await request.json()
    prompt = req.get('prompt', 'hi!')
    rid = req.get('rid', '0')
    output_length = req.get('max_length', 100)
    req = FloodRequest(rid=rid, input_text=prompt, output_length=output_length)
    output_queue = None
    for oq in output_queues:
        if oq.idle:
            output_queue = oq
            break
    if output_queue is None:
        raise ValueError("Output queues are empty!")
    output_queue.idle = False
    req.output_index = output_queue.index

    # results_generator = worker.async_stream_generate(req, input_queue, output_queue.queue)
    async def stream_results():
        async for output in worker.async_stream_generate(req, input_queue,
                                                         output_queue.queue):
            yield json.dumps(output, ensure_ascii=False).encode("utf-8")
        output_queue.idle = True

    return StreamingResponse(stream_results(), media_type="text/event-stream")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    global worker, input_queue, output_queues

    model_path = '/mntnlp/common_base_model/Qwen__Qwen2.5-7B-Instruct'


    n_stage = 1
    n_proc = 1
    eos_token_id = None
    worker = LLM(model_path,
                 #  model_dtype=torch.float8_e4m3fn,
                 #  head_dtype=torch.float8_e4m3fn,
                 #  emb_dtype=torch.float8_e4m3fn,
                 #  cache_dtype=torch.float8_e4m3fn,
                 n_stage=n_stage,
                 n_proc=n_proc,
                 cache_size=None,
                 slot_count=256,
                 schedule_mode='timely',
                 chunk_size=1024,
                 sync_wait_time=(4.0, 4.0),
                 queue_timeout=0.0005,
                 max_slot_alloc_fail_count=1,
                 alloc_early_exit_rate=0.95,
                 min_batch_size=32,
                 max_batch_size=256,
                 batch_size_step=128,
                 batch_size_round_frac=1.0,  # 0.585
                 min_decode_rate=0.97,  # 0.97
                 eos_token_id=eos_token_id,
                 kernels=('sa',),
                 spec_algo = 'lookahead',
                 spec_branch_length=8,
                 max_spec_branch_count=8,
                 logger='server.log',
                 debug=True)

    input_queue, chunk_queue, working_queue, output_queues = worker.initialize(
        output_queue_count=8)
    worker.launch(input_queue, chunk_queue, working_queue, output_queues)
    print("finish launch!")
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=30010,
        log_level='error',
        timeout_keep_alive=600,
        loop="uvloop",
    )
