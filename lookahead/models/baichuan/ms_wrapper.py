import os
import torch
from typing import Union, Dict, Any
from modelscope.pipelines.builder import PIPELINES
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from modelscope.pipelines.base import Pipeline
from modelscope.outputs import OutputKeys
from modelscope.pipelines.nlp.text_generation_pipeline import TextGenerationPipeline
from modelscope.models.base import Model, TorchModel
from modelscope.utils.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.utils import GenerationConfig


@PIPELINES.register_module(Tasks.text_generation, module_name='Baichuan2-7B-text-generation-pipe')
class Baichuan7BTextGenerationPipeline(TextGenerationPipeline):
    def __init__(
            self,
            model: Union[Model, str],
            *args,
            **kwargs):
        self.model = Baichuan7BTextGeneration(model) if isinstance(model, str) else model
        super().__init__(model=self.model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        output = {}
        device = self.model.model.device
        input_ids = self.model.tokenizer(inputs, return_tensors="pt").input_ids.to(device)
        pred = self.model.model.generate(input_ids, **forward_params)
        out = self.model.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        output['text'] = out
        return output

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@MODELS.register_module(Tasks.text_generation, module_name='Baichuan2-7B')
class Baichuan7BTextGeneration(TorchModel):
    def __init__(self, model_dir=None, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.logger = get_logger()
        # loading tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16,
                                                          trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_dir)
        self.model = self.model.eval()

    def forward(self, input: Dict, *args, **kwargs) -> Dict[str, Any]:
        output = {}
        response = self.model.chat(self.tokenizer, input)
        return {OutputKeys.RESPONSE: response, OutputKeys.HISTORY: ""}

    def quantize(self, bits: int):
        self.model = self.model.quantize(bits)
        return self

    def infer(self, input, **kwargs):
        device = self.model.device
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(device)
        pred = self.model.generate(input_ids, **kwargs)
        out = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return out
