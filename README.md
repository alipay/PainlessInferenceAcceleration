

<h1 align="center">Painless Inference Acceleration (PIA)</h1>


  
<p align="center">
   A toolkit to accelerate LLM inference without headache (🤯) and tears (😭) .
</p>


## NOTE

  [2025/03] We have transitioned our open-source license from the `Creative Commons Attribution 4.0 International` to the `MIT` license. This change reflects our assessment that the MIT License is more appropriate for the distribution and utilization of source code.

## *News or Update* 🔥

- [2025/03] We release our throughput-oriented inference framework [`FLOOD`](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/flood/README.md).

- [2024/01] We support all models of baichuan family (Baichuan-7b & 13b, Baichuan2-7b & 13b) for lookahead.

- [2024/01] We fully support repetition_penalty parameter for lookahead.

- [2024/01] We support Mistral & Mixtral for lookahead.

- [2023/12] We released our latency-oriented inference framework `LOOKAHEAD`.


## Introduction

Our repo, PIA (short for Painless Inference Acceleration), is designed for LLM inference and currently contains three key works:

- [`FLOOD`](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/flood/README.md): It employs pure pipeline parallelism to enhance inference throughput, thereby reducing communication costs typically associated with tensor parallelism. 

- [`LOOKAHEAD`](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/lookahead/README.md): It uses an on-the-fly trie-tree cache to prepare hierarchical multi-branch drafts, without the demand for assist models (e.g., speculative decoding) or additional head training (e.g., block decoding). 
With the efficient hierarchical structure, we can lookahead tens fo branches, therefore significantly improve generated tokens in a forward pass.

- [`IPAD`](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/ipad/README.md): It applies iterative pruning and distillation techniques to reduce the model size.


## Citations
```
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
```

```
@inproceedings{10.1145/3589335.3648321,
author = {Wang, Maolin and Zhao, Yao and Liu, Jiajia and Chen, Jingdong and Zhuang, Chenyi and Gu, Jinjie and Guo, Ruocheng and Zhao, Xiangyu},
title = {Large Multimodal Model Compression via Iterative Efficient Pruning and Distillation},
year = {2024},
isbn = {9798400701726},
publisher = {Association for Computing Machinery},
doi = {10.1145/3589335.3648321},
booktitle = {Companion Proceedings of the ACM Web Conference 2024},
pages = {235–244},
series = {WWW '24}
}
```