
from flood.models.modeling_llama_ignore import LlamaForCausalLM
from flood.models.modeling_bailing_ignore import BailingForCausalLM
from flood.models.modeling_bailing_moe_ignore import BailingMoeForCausalLM
from flood.models.modeling_qwen2_ignore import Qwen2ForCausalLM
from flood.models.modeling_glm import GLMForCausalLM
from flood.models.modeling_deepseek_ignore import DeepseekForCausalLM


model_class_map = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "BailingForCausalLM": BailingForCausalLM,
    "BailingMoeForCausalLM": BailingMoeForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "GLMForCausalLM": GLMForCausalLM,
    "DeepseekForCausalLM": DeepseekForCausalLM
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
    "GLMForCausalLM": ModelAttr(emb_name='glm.word_embeddings', layer_name='glm.transformer.layers', norm_name='glm.transformer.final_layernorm')
}