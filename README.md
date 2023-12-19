
<h1 align="center">Painless Inference Acceleration (PIA)</h1>


  
<p align="center">
   A toolkit for accelerating LLM inference without painness. Currently it only contains `lookahead`, a framework which accelerates LLM inference without loss of accuracy, other works will release soon.
</p>

## News or Update


## Performance Comparison

| model  | dataset         | GPU           | hugging face | Lookahead |
|---------------|---------------|---------------|-----------|-----------|
| Antglm-10b      | AntRAG-8k     | 1xA100-80G    | 52.4         |  280.9(x5.36)     |
| Llama2-chat-7b      | Dolly-15k      | 1xA100-80G    | 38.3         | 86.2(x2.25)     |
| Llama2-chat-13b      | Dolly-15k      | 1xA100-80G    | 34.0         | 71.7(x2.00)     |
| ChatGLM2-6b      | Dolly-15k      | 1xA100-80G    | 42.8         | 85.5(x2.11)     |


## Introduction

Our repo PIA (short for Painless Inference Acceleration) is used for LLM inference, it is based on transformers library of huggingface.co.

PIA includes the following modules:
- lookahead						# lookahead 
- lookahead/benchmarks          # benchmarks for several models, used for speed test and hyper-parameter tuning
- lookahead/common              # fundamental classes for lookahead
- lookahead/models			    # models supported by lookahead
- lookahead/examples            # minimum usage examples
- lookahead/tests			    # test cases
- requirements.txt


## Lincense （使用协议）

协议为CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

使用本项目前，请先阅读LICENSE.txt。如果您不同意该使用协议中列出的条款、法律免责声明和许可，您将不得使用本项目中的这些内容。



## Quick Start


Below is an example for the simplest use of `lookahead` to inference:

```python

import sys
import torch
from transformers import AutoTokenizer


sys.path.append('.') 
from common.lookahead_cache import LookaheadCache
from models.llama.modeling_llama import LlamaForCausalLM

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
git clone xxx
cd pia
pip install -r requirements.txt
cd lookahead/examples
python llama_example.py
```

To evaluation speedup of `lookahead`, we can run the scrips in the path `benchmarks/`,



## Customize Model

To support a customize model, usually we only need add a few lines, here is a example for supporting Llama:

```python

from common.pretrained_model import LookaheadPreTrainedModel
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
            # without lookahead
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            # embed positions
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
```


## Supported Models

We currently support a range of models, including Llama, OPT, Bloom, GPTJ, GPT2, Baichuan, ChatGLM, GLM, and Qwen. We welcome contributions to extend support to additional models.

## Tests

Tests can be run with:
```shell
pytest tests/ -s
```


## Citations