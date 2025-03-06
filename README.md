

<h1 align="center">Painless Inference Acceleration (PIA)</h1>


  
<p align="center">
   A toolkit to accelerate LLM inference without headache (🤯) and tears (😭) .
</p>


## NOTE

  [2025/03] We change our open-source license from `Creative Commons Attribution 4.0 International` to `MIT` license, as the latter is more suitable for source code. 

## *News or Update* 🔥

- [2025/03] We release our throughput-oriented inference framework `FLOOD`.

- [2024/01] We support all models of baichuan family (Baichuan-7b & 13b, Baichuan2-7b & 13b) for lookahead.

- [2024/01] We fully support repetition_penalty parameter for lookahead.

- [2024/01] We support Mistral & Mixtral for lookahead. [example](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/examples/mixtral_example.py) 

- [2023/12] We released our latency-oriented inference framework `LOOKAHEAD`.


## Introduction

Our repo PIA (short for Painless Inference Acceleration) is used for LLM inference, it contains three work currently:

- `LOOKAHEAD`: It uses an on-the-fly trie-tree cache to prepare hierarchical multi-branch drafts, without the demand for assist models (e.g., speculative decoding) or additional head training (e.g., block decoding). 
With the efficient hierarchical structure, we can lookahead tens fo branches, therefore significantly improve generated tokens in a forward pass.


- `FLOOD`: It uses pure pipeline parallelism to improve inference throughput.


## Citations

@inproceedings{10.1145/3637528.3671614,
author = {Zhao, Yao and Xie, Zhitian and Liang, Chen and Zhuang, Chenyi and Gu, Jinjie},
title = {Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
doi = {10.1145/3637528.3671614},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {6344–6355},
series = {KDD '24}
}
