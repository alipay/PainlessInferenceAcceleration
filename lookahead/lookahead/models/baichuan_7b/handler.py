import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# get dtype
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16


class EndpointHandler:
    def __init__(self, path=""):
        # load the model
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
        # create inference pipeline
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        # postprocess the prediction
        return prediction