from transformers import AutoTokenizer
import torch

from pia.lookahead.models.mixtral.modeling_mixtral import MixtralForCausalLM

model_dir = "/mntnlp/common_base_model/Mixtral-8x7B-Instruct-v0.1"
# config = AutoConfig.from_pretrained('/mntnlp/common_base_model/Mixtral-8x7B-Instruct-v0.1')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MixtralForCausalLM.from_pretrained(model_dir,
                                           trust_remote_code=False,
                                           low_cpu_mem_usage=True,
                                           torch_dtype=torch.float16,
                                           device_map="auto")



