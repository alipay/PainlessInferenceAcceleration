"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# from vllm import LLM, SamplingParams
from lookahead.common.lookahead_cache import LookaheadCache
from lookahead.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer


def main(args: argparse.Namespace):
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LlamaForCausalLM.from_pretrained(args.model
                                         , cache_dir='../'
                                         , torch_dtype=torch.float16
                                         , low_cpu_mem_usage=True
                                         , device_map='auto'
                                         )

    # sampling_params = SamplingParams(
    #     n=args.n,
    #     temperature=0.0 if args.use_beam_search else 1.0,
    #     top_p=1.0,
    #     use_beam_search=args.use_beam_search,
    #     ignore_eos=True,
    #     max_tokens=args.output_len,
    # )
    # print(sampling_params)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    stop_word_ids = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))
    # lookahead_cache = LookaheadCache(eos=tokenizer.eos_token_id, stop_words=stop_ids)
    # model.lookahead_cache = lookahead_cache
    
    prompt = "Hello, I'm am conscious and"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()
    position_ids = None

    debug_lookahead = False
    decoding_length = 63
    branch_length = 12
    max_new_tokens = args.output_len
    decoding_kwargs = {"use_lookahead": True,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length,
                       "stop_word_ids": stop_word_ids}

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                decoding_kwargs = {"use_lookahead": True,
                    "debug_lookahead": debug_lookahead,
                    "decoding_mode": 'hier',
                    "decoding_length": decoding_length,
                    "branch_length": branch_length,
                    "stop_word_ids": stop_word_ids}
                llm.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             decoding_kwargs=decoding_kwargs
                             )
            print(p.key_averages())
        else:
            decoding_kwargs = {"use_lookahead": True,
                                "debug_lookahead": debug_lookahead,
                                "decoding_mode": 'hier',
                                "decoding_length": decoding_length,
                                "branch_length": branch_length,
                                "stop_word_ids": stop_word_ids}
            start_time = time.perf_counter()
            llm.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=1.0,
                        do_sample=False,
                        decoding_kwargs=decoding_kwargs
                        )
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(
                "") / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=args.profile_result_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    print(f'Avg latency: {np.mean(latencies)} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='/mntnlp/common_base_model/llama2-7b')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=(
            'path to save the pytorch profiler output. Can be visualized '
            'with ui.perfetto.dev or Tensorboard.'
        ))
    args = parser.parse_args()
    main(args)