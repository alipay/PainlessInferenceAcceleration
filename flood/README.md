

<h1 align="center">FLOOD</h1>


  
<p align="center">
   FLOOD, a throughput-oriented framework with pipeline parallism and segment cache.
</p>


## *News or Update* 🔥

- [2025/03] We release our code of our inference framework `FLODD`.



## Models we support 

- Ling MoE
- Ling
- Llama
- Qwen
- Deepseek v1

## Roadmap

1. Integrate our previous work `LOOKAHEAD`. 

2. Improve prefill performance with Prefix caching.

3. Improve performance with CUDA-Graph.

4. Implement segment attention with `CUTE` for better performance, especially with FP8 kvcache.

5. Support more models, include Deepseek R1, etc.


## Performance Comparison


### Throughput

Performance is measured by token/s(tokens per second) of generated tokens. The version of vLLM is 0.6.6.post2, we enable the chunk prefill with chunk size 2048, other parameters are the same as default.


| model    | dataset     | GPU      | vLLM | flood    | speedup |
|-------------|-------------|----------|-----------------|--------------|
| Llama3-8B   | shareGPT  | 1*A100 |     3201    | 4529 | 1.41 |
| Ling-Lite | shareGPT|  1 * H20  |  4355 | 5869 | 1.35 |
| Ling-Lite | shareGPT|  1 * A100 | 3576 | 5451 | 1.52 |
| Ling-Plus(FP8)| shareGPT | 8 * H20 | 2742 | 6569 | 2.40 |

### Kernels

Performance is measured by TFLOPS. Attention head number is 64, kv head number is 8 and kv head dimension is 128. More detail can be found in benchmark/bench_seg_attn.py.

| Device | BatchSize   |  Q_len    | K_len      | flash-attn | seg-attn    | speedup |
|-------------|----------|-------------|----------|-----------------|--------------|
|A100|  1  | 1024  | 1024 |     99.19    | 107.35 | 1.08 |
|A100 | 128 | 1|  1024  |  10.65 | 13.56  | 1.27 |
|H20|  1  | 1024  | 1024 |   90.28      | 96.05 | 1.06 |
|H20 | 128 | 1|  1024  | 7.16  |  22.63 | 3.16 |

## Introduction

Flood is an efficient offline inference framework, which is entirely based on pipeline parallelism. It also uses segmentable block instead of paged blocks to  
manage kvcache, to maximum the continuity of the kvcache of a request. 
We implement an attention kernel, named SegmentAttention, to cooperate with the segmentable kvcache. Flood supports the following features currently:
1. chunk prefill
2. quantization(fp8/int8)
3. multi-node inference 
4. stream inference 
5. multiple scheduling strategies
6. PPL evaluation
7. multi-modal model inference
8. sampling



## Lincense

This code repository is released under the MIT License.

## Installation

1. Clone this repository and navigate to PainlessInferenceAcceleration
```
git clone https://github.com/alipay/PainlessInferenceAcceleration.git
cd PainlessInferenceAcceleration/flood
```
2. Install Package
```
python setup.py install
```

### requirements 

We mainly develop and benchmark on the environment below, lower version may also be OK.

cuda>=12.4 (higher is better)
torch>=2.5.0 (higher is better)
triton>=3.1.0 (higher is better) 
accelerate>=1.4.0
transformers>=4.47.1
flash-attn>=2.6.3 is required if use `fa2` kernel 
flash-attn-3>=3.0.0 is required if use `fa3` kernel
vllm>=0.6.2 is required if use int8 quantization



## Quick Start

A simple example can be found in example/simple_example.py.

To reproduce the reported performance, run the benchmark/bench_flood.py.


## AKKNOWLEDGE

Flood is inspired by FlashAttention 2&3, vLLM, flashinfer projects.


## Citations

Please wait a moment.

## Contact Us
For technical questions and feature requests, please use Github issues or discussions.

