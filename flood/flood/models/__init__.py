
from flood.models.modeling_llama import LlamaForCausalLM
from flood.models.modeling_bailing import BailingForCausalLM
from flood.models.modeling_bailing_moe import BailingMoeForCausalLM
from flood.models.modeling_qwen2 import Qwen2ForCausalLM
from flood.models.modeling_deepseek import DeepseekForCausalLM
from flood.models.modeling_deepseekv3 import DeepseekV3ForCausalLM


model_class_map = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "BailingForCausalLM": BailingForCausalLM,
    "BailingMoeForCausalLM": BailingMoeForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "DeepseekForCausalLM": DeepseekForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM
}


# used for device map setting
class ModelAttr:
    def __init__(self, emb_name='model.embed_tokens', layer_name='model.layers', norm_name='model.norm', head_name="lm_head"):
        self.emb_name = emb_name 
        self.layer_name = layer_name 
        self.norm_name = norm_name 
        self.head_name = head_name 

model_attr_map = {
    "DEFAULT": ModelAttr(),
    "BailingForCausalLM": ModelAttr(emb_name='model.word_embeddings'),
    "BailingMoeForCausalLM": ModelAttr(emb_name='model.word_embeddings'),
}