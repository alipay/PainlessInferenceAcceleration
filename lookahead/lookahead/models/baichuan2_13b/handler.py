import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.generation.utils import GenerationConfig

# get dtype
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16

class EndpointHandler:
    def __init__(self, path=""):
        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)

    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        inputs = data.pop("inputs", data)
        # ignoring parameters! Default to configs in generation_config.json.
        messages = [{"role": "user", "content": inputs}]
        response = self.model.chat(self.tokenizer, messages)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return [{'generated_text': response}]
