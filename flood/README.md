

<h1 align="center">FLOOD</h1>


  
<p align="center">
   FLOOD, a throughput-oriented framework with pipeline parallism and segment cache.
</p>


## *News or Update* 🔥

- [2025/03] We release our code of our inference framework `FLODD`.



## Models we support 

- Ling MoE model
- Ling dense model
- Llama
- Qwen
- Deepseek v1

## Roadmap

1. Integrate our previous work `LOOKAHEAD`. 

2. Improve prefill performance with Prefix caching.

3. Improve performance with CUDA-Graph.

4. Implement segment attention with `CUTE`, for better performance, especially with FP8 kvcache.


## Performance Comparison

Performance is measured by token/s(tokens per second) of generated tokens.

### Public datasets and models


| model                  | dataset     | GPU      | vLLM | flood    | speedup |
|------------------------|-------------|----------|-----------------|--------------|
| Llama3-8B   | shareGPT  | 1*A100 |     3201    | 4529 | 1.41 |
| Ling-Lite | shareGPT|  1 * H20  |  4355 | 5869 | 1.35 |
| Ling-Lite | shareGPT|  1 * A100 | 3576 | 5451 | 1.52 |
| Ling-Plus(FP8)| shareGPT | 8 * H20 | 2742 | 6569 | 2.40 |


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

## Quick Start

A simple example can be found in example/simple_example.py.

To reproduce the reported performance, you can run the benchmark/bench_flood.py.


## AKKNOWLEDGE

Flood is inspired by FlashAttention 2&3, vLLM, flashinfer projects.


## Citations




## Contact Us
For technical questions and feature requests, please use Github issues or discussions.

