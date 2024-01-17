from transformers import AutoTokenizer
import torch

from pia.lookahead.models.mistral.modeling_mistral import MistralForCausalLM

model_dir = "/mntnlp/chengle/mistralai__Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MistralForCausalLM.from_pretrained(model_dir,
                                           trust_remote_code=False,
                                           low_cpu_mem_usage=True,
                                           torch_dtype=torch.float16,
                                           device_map="auto")

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
inputs['input_ids'] = inputs.input_ids.cuda()
inputs['attention_mask'] = inputs.attention_mask.cuda()
position_ids = None


model(**inputs)

