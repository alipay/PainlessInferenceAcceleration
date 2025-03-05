

<h1 align="center">LOOKAHEAD</h1>


  
<p align="center">
   LOOKAHEAD, a framework which accelerates LLM inference without loss of accuracy.
</p>

<!-- [[Paper](https://arxiv.org/abs/2312.12728)] -->


## *News or Update* 🔥

- [2024/01] We support all models of baichuan family (Baichuan-7b & 13b, Baichuan2-7b & 13b).

- [2024/01] We fully support repetition_penalty parameter.

- [2024/01] We support Mistral & Mixtral. [example](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/examples/mixtral_example.py)

- [2023/12] We released our [Lookahead paper](https://arxiv.org/abs/2312.12728) on arXiv!



## Models we support 

- GLM
- Baichuan & Baichuan 2 
- BLOOM
- ChatGLM 2 & 3 
- GPT-2
- GPT-J
- InterLM
- LLaMA & LLaMA-2
- Mistral
- Mixtral 
- OPT
- Qwen

## Known issuss & TODO

<del> ISSUE 1. Repetition_penalty is not fully supported, we will fix it in the future.  </del>

ISSUE 2. Lookahead may generate responses different from original ones due to low-precise data type (i.e., fp16 or bf16), the responses would be the same with fp32.

ISSUE 3. Baichuan tokenizer cannot be initialized with the lastest version transformers (4.30.2 can work).

ISSUE 4. Qwen model may generate slightly different responses with lookahead when the repetition_penalty parameter is set.

<del> TODO1: Support the latest version  [🤗 transformers](https://github.com/huggingface/transformers) ]. Currently it's based on 4.30.2. </del>

TODO2: Integrate our work [FastCoT](https://arxiv.org/pdf/2311.08263.pdf)

TODO3: Optimize batch inference implementation with flash-attention.


## Performance Comparison

Performance is measured by token/s(tokens per second) of generation tokens.

### Public datasets and models

We use the test set for evaluation and the train set for trie-tree cache construction. The hyper-parameters are tuned by grid searching. The tag `fused` indicates operators are fused with triton, the implementation can be found in `modeling_llama_batch.py`.

| model                  | dataset     | GPU      | 🤗 transformers | lookahead    |
|------------------------|-------------|----------|-----------------|--------------|
| Llama2-7b-chat         | Dolly-15k   | A100-80G | 40.6            | 83.7 (x2.06)  |
| Llama2-7b-chat         | GSM-8k      | A100-80G | 41.4            | 111.3 (x2.69) |
| Llama2-7b-chat(fused)  | Dolly-15k   | A100-80G | 50.4            | 106.8 (x2.12) |
| Llama2-7b-chat(fused)  | Dolly-15k   | A10      | 31.4            | 55.7(x1.77) |
| Llama2-7b-chat(fused)  | GSM-8k      | A100-80G | 53.7            | 149.6 (x2.79) |
| Llama2-7b-chat(fused)  | GSM-8k      | A10      | 31.4            | 68.1(x2.17) |
| Llama2-7b-chat(fused)  | Humaneval-x | A100-80G | 51.1             | 161.5(x3.16) |
| Llama2-7b-chat(fused)  | Humaneval-x      | A10      | 30.9            | 89.6(x2.90) |
| Llama2-13b-chat        | Dolly-15k   | A100-80G | 34.0            | 71.7 (x2.11)  |
| Llama2-13b-chat        | GSM-8k      | A100-80G | 31.2            | 71.1 (x2.28)  |
| Llama2-13b-chat(fused) | Dolly-15k   | A100-80G | 39.9            | 84.6 (x2.12)  |
| Llama2-13b-chat(fused) | Dolly-15k   | V100-32G | 20.5            | 35.2(x1.72)  |
| Llama2-13b-chat(fused) | GSM-8k      | A100-80G | 42.9            | 103.4 (x2.41) |
| Llama2-13b-chat(fused) | GSM-8k      | V100-32G | 22.0            | 45.6(x2.07) |
| Llama2-13b-chat(fused) | Humaneval-x   | A100-80G | 35.0            | 137.3(x3.92)  |
| Llama2-13b-chat(fused) | Humaneval-x      | V100-32G | 21.5            | 57.0(x2.65) |
| ChatGLM2-6b            | Dolly-15k   | A100-80G | 45.6            | 108.4 (x2.38) |
| ChatGLM2-6b            | GSM-8k      | A100-80G | 43.3            | 94.0 (x2.17)  |


We test 5 examples with Llama2-7b-chat and dolly dataset, inference time without lookahead (the left figure) is 15.7s (48.2token/s), while inference time with lookahead is 6.4s (112.9token/s), speedup is 2.34.

[//]: # (![glm_without_lookahead]&#40;./pia/lookahead/figures/llama_la_off.gif&#41;![glm_with_lookahead]&#40;./pia/lookahead/figures/llama_la_on.gif&#41;)
[//]: # (<div align=center>)

[//]: # (<img src="./pia/lookahead/figures/llama_la_off.gif" width="50%"><img src="./pia/lookahead/figures/llama_la_on.gif" width="50%">)

[//]: # (</div>)
<div align=center>
<img src="https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/llama_la_off.gif" width="50%"><img src="https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/llama_la_on.gif" width="50%">
</div>



### Private datasets and models

We use the first 1000 samples for evaluation and the rest for trie-tree cache construction. The hyper-parameters are `decoding_length=128` and `branch_lenght=32`.

Our method could obtain significant acceleration in RAG (Retrieval Augmented Generation) scenarios. However, there is no real-life datasets available currently. Therefore, we only evaluate on our private datasets and models. 
AntGLM-10B is a LLM developed by Ant Group with [GLM](https://huggingface.co/THUDM/glm-10b-chinese) architecture. 

| model          | scenarios       | GPU | 🤗 transformers | Lookahead    |
|----------------|---------------|--|-----------------|--------------|
| AntGLM-10b     | Citizen Biz Agent     | A100-80G | 52.4            | 280.9(x5.36) |
| AntGLM-10b     | Citizen Biz Agent     | A10 | 20.3            | 105.1(x5.18) |
| AntGLM-10b     | Citizen Biz Agent     | V100-32G | 27.3            | 118.9(x4.36) |
| AntGLM-10b     | Enterprise Info QA    | A100-80G | 50.7            | 259.1(x5.11) |
| AntGLM-10b     | Health Suggestion     | A100-80G | 51.6            | 240.2(x4.66) |


[//]: # (![llama_without_lookahead]&#40;./pia/lookahead/figures/glm_la_off.gif&#41;![llama_with_lookahead]&#40;./pia/lookahead/figures/glm_la_on.gif&#41;)

We test 5 examples with AntGLM-10B and AntRag dataset, inference time without lookahead (the left figure) is 16.9s (33.8token/s), while inference time with lookahead is 3.9s (147.6token/s), speedup is 4.37.

[//]: # (<div align=center>)

[//]: # (<img src="./pia/lookahead/figures/glm_la_off.gif" width="50%"><img src="./pia/lookahead/figures/glm_la_on.gif" width="50%">)

[//]: # (</div>)

<div align=center>
<img src="https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/glm_la_off.gif" width="50%"><img src="https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/glm_la_on.gif" width="50%">
</div>

## Introduction

Our repo PIA (short for Painless Inference Acceleration) is used for LLM inference, it is based on [🤗 transformers](https://github.com/huggingface/transformers)  library.

- It uses an on-the-fly trie-tree cache to prepare hierarchical multi-branch drafts, without the demand for assist models (e.g., speculative decoding) or additional head training (e.g., block decoding). 
With the efficient hierarchical structure, we can lookahead tens fo branches, therefore significantly improve generated tokens in a forward pass.

- You can also benefit from our optimized fuesed operation kernels.

Note that our work is different from the other method named [lookahead decoding](https://github.com/hao-ai-lab/LookaheadDecoding). 


### Hierarchical multi-branch draft


![workflow](lookahead/figures/flow.png)


![mask](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/dynamic.gif)


![construction](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/trie_construct.gif)


![retrieve](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/pia/lookahead/figures/trie_retrieve.gif)


## Lincense

CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

## Installation

1. Clone this repository and navigate to PainlessInferenceAcceleration
```
git clone https://github.com/alipay/PainlessInferenceAcceleration.git
cd PainlessInferenceAcceleration
```
2. Install Package
```
python setup.py install
```

## Quick Start


Below is an example for the simplest use of `lookahead` to inference:

```python

import torch
from transformers import AutoTokenizer


from lookahead.common.lookahead_cache import LookaheadCache
from lookahead.models.llama.modeling_llama import LlamaForCausalLM

model_dir = 'meta-llama/Llama-2-7b-chat-hf'
model = LlamaForCausalLM.from_pretrained(model_dir
                                         , cache_dir='./'
                                         , torch_dtype=torch.float16
                                         , low_cpu_mem_usage=True
                                         , device_map='auto'
                                         )
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")

output_ids = model.generate(input_ids=inputs.input_ids.cuda(),
                            attention_mask=inputs.attention_mask.cuda(),
                            max_new_tokens=256,
                            decoding_kwargs={'use_lookahead': True}
                            )
response = tokenizer.decode(output_ids[0].tolist())
print(f'{response=}')
```

To use `lookahead` with other models, we can run the scripts in the path `examples/`.
Each supported models are included and  can be used for correctness evaluation.

```shell
python [model name]_example.py
```

## Benchmarks

To evaluation speedup of `lookahead`, we can run the scripts in the path `benchmarks/`, the preprocess of datasets can be found in `benchmarks/preprocess_sample.py`. 

To inspect running details of lookahead, we can turn on `return_dict_in_generate`, i.e.,

```python
outputs = model.generate(...,
                        return_dict_in_generate=True
                        )
output_ids = outputs.sequences
kwargs = outputs.kwargs
# edls: short for effective decoding lengths, i.e., generate token count in a forward, therefore edls always >=1 ( even without lookahead, we will generate one token in a forward, so edls=1)
edls = kwargs['edls']
# dls: short of decoding lengths, i.e., token count in a forward, always >= 1. Note that it is set to 1 intead of prompt length in the prefill stage.
dls = kwargs['dls']
# fts: short for forward time(s), the first is the prefill time and others are decoding times.
fts = kwargs['fts']
#qts: short of query time(s), i.e., the time for retrieving a sub trie tree.
qts = kwargs['qts']
```


## Customize Model

<details>

<summary>To support a customize model, usually we only need add a few lines, here is a example for supporting Llama: </summary>

```python

from lookahead.common.pretrained_model import LookaheadPreTrainedModel
class LlamaPreTrainedModel(LookaheadPreTrainedModel):
    '''
    other code
    '''

class LlamaModel(LlamaPreTrainedModel):

    '''
    other code
    '''

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        '''
        other code
        '''

        """
        NOTE: adapt for lookahead
        lookahead always use a rank-4 tensor for attention_mask, then a minimum adaption for lookahead is routed by the rank,
        Lookahead: generate position_ids from attention_masks and set zero elements of the mask to -inf 
        """
        if attention_mask is not None and len(attention_mask.shape) == 4:
            # with lookahead
            position_ids = torch.sum(attention_mask, dim=-1).squeeze(1) - 1
            attention_mask = (1.0-attention_mask.to(inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
        else:
            # without lookahead, reuse the original code lines
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
```

Note that the above adaption can not be used for batch inference, as generated token length of different samples may be varied. Adaption for batch 
inference can be found in `models/modeling_glm_batch.py` or `models/modeling_llama_batch.py`. `Flash-attention` enhanced batch inference is on developing.

</details>


<!-- ## Supported Models

We currently support a range of models, including Llama, OPT, Bloom, GPTJ, GPT2, Baichuan, ChatGLM, GLM, and Qwen. We welcome contributions to extend support to additional models.  -->

## Tests

Tests can be run with:
```shell
cd lookahead
pytest tests/ -s
```


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

