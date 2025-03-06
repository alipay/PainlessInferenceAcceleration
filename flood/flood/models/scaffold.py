# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


class Scaffold():
    def __init__(self,
                 model_name='Llama',
                 emb_name='embed_tokens',
                 head_name='lm_head',
                 attn_name='self_attn',
                 mlp_name='mlp',
                 mlp_norm_name='input_layernorm',
                 attn_norm_name='post_attention_layernorm',
                 final_norm_name='norm',
                 norm_type='rms',
                 qkv_proj_name='q_proj,k_proj,v_proj',
                 o_proj_name='o_proj',
                 gate_up_proj_name='gate_proj,up_proj',
                 down_proj_name='down_proj',
                 act_type='silu_and_mul',
                 eps_name='rms_norm_eps',
                 num_layers_name='num_hidden_layers',
                 use_bias=(False, False, False, False),
                 moe=False,
                 num_experts_name='num_experts',
                 num_shared_experts_name='num_shared_experts'
                 ):
        self.model_name = model_name
        self.emb_name = emb_name
        self.head_name = head_name
        self.attn_name = attn_name
        self.mlp_name = mlp_name
        self.mlp_norm_name = mlp_norm_name
        self.attn_norm_name = attn_norm_name
        self.final_norm_name = final_norm_name
        self.norm_type = norm_type
        self.qkv_proj_name = qkv_proj_name
        self.o_proj_name = o_proj_name
        self.gate_up_proj_name = gate_up_proj_name
        self.down_proj_name = down_proj_name
        self.act_type = act_type
        self.eps_name = eps_name
        self.num_layers_name = num_layers_name
        self.use_bias = use_bias
        self.moe = moe
        self.num_experts_name = num_experts_name
        self.num_shared_experts_name = num_shared_experts_name

    def generate_import(self):
        if self.model_name in ('Bailing', 'BailingMoe', 'Deepseek'):
            mn = self.model_name.lower().replace('moe','_moe')
            conf = f"from .configuration_{mn} import {self.model_name}Config"
        else:
            conf = f"from transformers.models.{self.model_name.lower()}.configuration_{self.model_name.lower()} import {self.model_name}Config"

        if self.moe:
            moe_line = """from flood.layers.moe import AutoExperts"""
        else:
            moe_line = ''

        line = f"""# -*- coding: utf-8 -*-
# Copyright (c) Ant Financial Service Group and its affiliates.

# NOTE: the file is generated by scaffold.py

import os
import math
import time
import copy
from typing import List, Optional, Tuple, Union, Dict

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

from flood.ops import RMSNorm, silu_and_mul
from flood.utils.batch import Batch
from flood.layers.linear import AutoLinear
from flood.layers.rope import AutoRope
from flood.layers.attention import AutoAttention
from flood.layers.embedding import AutoEmbedding
from flood.layers.sampler import Sampler
{moe_line}
{conf}

"""
        return line

    def generate_mlp(self):
        inner_line = '_' if self.moe else ''

        if ',' in self.gate_up_proj_name:
            gate_name, up_name = self.gate_up_proj_name.split(',')
            gate_up_name = 'gate_up_proj'

            gate_up_linear = f"""
        self.{gate_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                    self.intermediate_size, 
                                                    bias={self.use_bias[2]}, 
                                                    config=config, 
                                                    name='{gate_name}')
        self.{up_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                  self.intermediate_size, 
                                                  bias={self.use_bias[2]}, 
                                                  config=config, 
                                                  name='{up_name}')
            """

            gate_up_patch = f"""
        self.{gate_up_name} = self.{gate_name}.merge([self.{gate_name}, self.{up_name}])
        self.{self.down_proj_name}.patch()

        self.{gate_name} = None
        delattr(self, '{gate_name}')

        self.{up_name} = None
        delattr(self, '{up_name}')
            """

        else:
            gate_up_name = self.gate_up_proj_name
            gate_up_linear = f"""
        self.{gate_up_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                    2 * self.intermediate_size, 
                                                    bias={self.use_bias[2]}, 
                                                    config=config, 
                                                    name='{gate_up_name}')
            """
            gate_up_patch = ''
        act = self.get_act_line()
        line = f"""
class {self.model_name}MLP(torch.nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
{gate_up_linear}
        self.{self.down_proj_name} = AutoLinear.from_pretrained(self.intermediate_size, 
                                                    self.hidden_size, 
                                                    bias={self.use_bias[3]}, 
                                                    config=config, 
                                                    name='{self.down_proj_name}')

    def {inner_line}flood_patch_func(self, kwargs=None):
        if self.layer_idx == 0:
            print('patch MLP')
{gate_up_patch}
    def forward(self, x):
        gate_up = self.{gate_up_name}(x)
        act = {act}(gate_up)
        return self.{self.down_proj_name}(act)
"""
        return line


    def generate_moe(self):
        if not self.moe:
            return ''
        line = f"""
class {self.model_name}MoE(torch.nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int = 0
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.{self.num_experts_name} = config.{self.num_experts_name}
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = self.config.norm_topk_prob

        exp_conf = copy.deepcopy(config)
        exp_conf.intermediate_size = config.moe_intermediate_size

        modules = torch.nn.ModuleList([{self.model_name}MLP(exp_conf, layer_idx=-1)
                                        for _ in range(self.{self.num_experts_name})])
        self.experts = AutoExperts.from_pretrained(module_list=modules,
                                                    hidden_size=exp_conf.hidden_size,
                                                    intermediate_size=exp_conf.intermediate_size,
                                                    num_expert=self.{self.num_experts_name},
                                                    config=config
                                                    )

        self.gate = torch.nn.Linear(config.hidden_size,
                                     self.{self.num_experts_name},
                                     bias=False)

        im_sz = config.moe_intermediate_size * config.{self.num_shared_experts_name}
        share_conf = copy.deepcopy(config)
        share_conf.intermediate_size = im_sz
        self.shared_experts = {self.model_name}MLP(config=share_conf, layer_idx=-1)

    def flood_patch_func(self, kwargs=None):
        self.experts._flood_patch_func()

        self.shared_experts._flood_patch_func()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = self.shared_experts(hidden_states)

        router_logits = self.gate(hidden_states)

        final_hidden_states = self.experts(hidden_states,
                                        router_logits,
                                        self.top_k,
                                        renormalize=self.norm_topk_prob
                                        )
        final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)
"""
        return line


    def generate_attn(self):
        attn = self.generate_attn_class()

        forward = self.generate_attn_forward()

        return attn + '\n' + forward

    def generate_attn_class(self):

        if ',' in self.qkv_proj_name:
            qkv_name = 'qkv_proj'
            q_name, k_name, v_name = self.qkv_proj_name.split(',')
            qkv_linear = f"""
        self.{q_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                 self.num_heads * self.head_dim, 
                                                 bias={self.use_bias[0]}, 
                                                 config=config, 
                                                 name='{q_name}')
        self.{k_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                 self.num_key_value_heads * self.head_dim, 
                                                 bias={self.use_bias[0]}, 
                                                 config=config, 
                                                 name='{k_name}')
        self.{v_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                 self.num_key_value_heads * self.head_dim, 
                                                 bias={self.use_bias[0]}, 
                                                 config=config, 
                                                 name='{v_name}')
            """
            qkv_patch = f"""

        self.{qkv_name} = self.q_proj.merge([self.{q_name}, self.{k_name}, self.{v_name}])

        self.{self.o_proj_name}.patch()

        self.{q_name} = None
        delattr(self, '{q_name}')

        self.{k_name} = None
        delattr(self, '{k_name}')

        self.{v_name} = None
        delattr(self, '{v_name}')
"""

        else:
            qkv_name = self.qkv_proj_name
            qkv_linear = f"""
        qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.{qkv_name} = AutoLinear.from_pretrained(self.hidden_size, 
                                                 qkv_dim, 
                                                 bias={self.use_bias[0]}, 
                                                 config=config, 
                                                 name='{qkv_name}')
"""
            qkv_patch = ''

        line = f"""
class {self.model_name}Attention(torch.nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        head_dim = None
        if hasattr(config, 'head_dim'):
            head_dim = config.head_dim
        if head_dim is None or head_dim<=0:
            head_dim = self.hidden_size // self.num_heads
        self.head_dim = head_dim
        self.intermediate_size = self.num_heads * self.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = float(config.rope_theta)
        self.softmax_scale = math.sqrt(1.0 / self.head_dim)
{qkv_linear}
        self.{self.o_proj_name} = AutoLinear.from_pretrained(self.intermediate_size, 
                                                 self.hidden_size, 
                                                 bias={self.use_bias[1]}, 
                                                 config=config, 
                                                 name='{self.o_proj_name}')

        self.rope = AutoRope.from_pretrained(config)
        self.attention =  None

    def flood_patch_func(self, kwargs=None):
        if self.layer_idx == 0:
            print('patch Attention')
{qkv_patch}
        if kwargs is None:
            kwargs = {{}}
        cache_dtype = kwargs.get('cache_dtype', None)
        interleave_value = kwargs.get('interleave_value', False)
        if interleave_value:
            AutoAttention.interleave(self.{qkv_name}, 
                                     self.num_heads, 
                                     self.num_key_value_heads, 
                                     self.head_dim)

        kernels = kwargs.get('kernels', ['sa'])
        self.attention = AutoAttention.from_pretrained(cache_dtype, 
                                                       layer_idx=self.layer_idx, 
                                                       kernels=kernels,
                                                       softmax_scale=self.softmax_scale)
"""
        return line


    def generate_attn_forward(self):

        qkv_name = 'qkv_proj' if ',' in self.qkv_proj_name else self.qkv_proj_name

        line = f"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:

        q_len = hidden_states.size(0)
        qkv = self.{qkv_name}(hidden_states)
        qkv = qkv.view(q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split([self.num_heads, 
                                                            self.num_key_value_heads, 
                                                            self.num_key_value_heads], 
                                                            dim=-2)

        batch_meta_info = kwargs['batch_meta_info']

        self.rope(query_states, key_states, batch_meta_info.q_offsets, batch_meta_info.pids)

        attn_output = self.attention(query_states, key_states, value_states, 
                                     batch_meta_info, past_key_value)

        # model may have different hidden_size
        attn_output = attn_output.view(q_len, self.intermediate_size)  
        attn_output = self.{self.o_proj_name}(attn_output)

        return attn_output
"""
        return line

    def generate_layer(self):

        norm_line = self.get_norm_line()
        mlp_name = self.model_name + ('MoE' if self.moe else 'MLP')
        line = f"""
class {self.model_name}DecoderLayer(torch.nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # use layer_idx==None to indicate that the layer does not 
        # initialized on the current node, and use layer_idx==-1
        # to indicate the final layer of the model
        if self.layer_idx is not None:
            self.{self.attn_name} = {self.model_name}Attention(config, layer_idx=layer_idx)
            self.{self.mlp_name} = {mlp_name}(config, layer_idx=layer_idx)
            self.{self.attn_norm_name} = {norm_line}
            self.{self.mlp_norm_name} = {norm_line}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_meta_info: Optional[Batch] = None,
    ) -> torch.Tensor:

        if self.layer_idx is None:
            return hidden_states

        residual = hidden_states

        hidden_states = self.{self.attn_norm_name}(hidden_states)

        hidden_states = self.{self.attn_name}(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_meta_info=batch_meta_info
        )

        hidden_states += residual

        if self.layer_idx == -1 and batch_meta_info.logit_indices is not None:
            if batch_meta_info.logit_indices.numel() == 0:
                return
            hidden_states = hidden_states[batch_meta_info.logit_indices]

        residual = hidden_states
        hidden_states = self.{self.mlp_norm_name}(hidden_states)
        hidden_states = self.{self.mlp_name}(hidden_states)

        hidden_states += residual

        return hidden_states
"""
        return line

    def generate_model(self):

        norm_line = self.get_norm_line()

        line = f"""
class {self.model_name}Model(PreTrainedModel):

    config_class = {self.model_name}Config
    base_model_prefix = "model"
    _no_split_modules = ["{self.model_name}DecoderLayer"]

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))

        if self.rank == 0:
            self.{self.emb_name} = AutoEmbedding.from_pretrained(config, 
                                                                config.vocab_size, 
                                                                config.hidden_size, 
                                                                padding_idx=self.padding_idx)
        else:
            self.{self.emb_name} = None 

        n_layer = config.{self.num_layers_name}
        layers = []
        local_size = n_layer // self.world_size
        for i in range(n_layer):
            layer_idx = i if i // local_size == self.rank else None
            layer_idx = -1 if layer_idx == n_layer - 1 and self.rank == self.world_size - 1 else layer_idx
            layers.append({self.model_name}DecoderLayer(config, layer_idx=layer_idx))
        self.layers = torch.nn.ModuleList(layers)

        self.{self.final_norm_name} = {norm_line}

    def get_input_embeddings(self):
        return self.{self.emb_name}

    def set_input_embeddings(self, value):
        self.{self.emb_name} = value
"""

        return line

    def generate_facade(self):

        if self.model_name == 'Bailing':
            norm_head_line = """
    def flood_patch_func(self, kwargs=None):
        if hasattr(self.config, 'norm_head') and self.config.norm_head:
            dtype = self.lm_head.weight.data.dtype
            norm = torch.norm(self.lm_head.weight.data.float(), 
                              p=2, dim=0, keepdim=True) + 1e-7
            self.lm_head.weight.data /= norm.to(dtype)
        if hasattr(self.lm_head, 'patch'):
            self.lm_head.patch()
"""
        else:
            norm_head_line = """
    def flood_patch_func(self, kwargs=None):
        if hasattr(self.lm_head, 'patch'):
            print('patch lm_head')            
            self.lm_head.patch()
"""

        line = f"""
class {self.model_name}ForCausalLM(PreTrainedModel):
    config_class = {self.model_name}Config
    _tied_weights_keys = ["{self.head_name}.weight"]

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.model = {self.model_name}Model(config)
        self.vocab_size = config.vocab_size

        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if self.rank == self.world_size - 1:
            self.{self.head_name} = AutoLinear.from_pretrained(config.hidden_size, 
                                                    config.vocab_size, 
                                                    bias=False, 
                                                    config=config, 
                                                    name='lm_head')        
        else:
            self.{self.head_name} = None
        self.sampler = Sampler()


    def get_input_embeddings(self):
        return self.model.{self.emb_name}

    def set_input_embeddings(self, value):
        self.model.{self.emb_name} = value

    def get_output_embeddings(self):
        return self.{self.head_name}

    def set_output_embeddings(self, new_embeddings):
        self.{self.head_name} = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

{norm_head_line}

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_meta_info : Batch = None,
        device_list : List = None,
        sync_layers : List = None,
        streams : List = None
    ) -> List:

        n_devices = len(device_list)
        n_layers = len(self.model.layers)
        for i, indices in enumerate(device_list):
            stream = streams[i]
            with torch.cuda.stream(stream):
                if i == 0 and self.rank == 0:
                    batch_meta_info.to(torch.device(0), non_blocking=True)
                    hidden_states = self.model.{self.emb_name}(batch_meta_info.input_ids)
                    embeddings = batch_meta_info.embeddings
                    if embeddings is not None:
                        emb_idx_list = batch_meta_info.emb_idx_list
                        for ie, emb_idx in enumerate(emb_idx_list):
                            if emb_idx is None:
                                continue
                            ss, se, ds, de = emb_idx
                            hidden_states[ds:de] = embeddings[ie][ss:se]
                sync_layers[i]()

                for j in indices:
                    hidden_states = self.model.layers[j](
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        batch_meta_info=batch_meta_info,
                    )

                if i < n_devices-1:
                    device = torch.device(i + 1)
                    hidden_states = hidden_states.to(device, non_blocking=True)
                    batch_meta_info.to(device, non_blocking=True)
                else:
                    if self.rank == self.world_size - 1 and hidden_states is not None:
                        hidden_states = self.model.{self.final_norm_name}(hidden_states)
                        logits = self.{self.head_name}(hidden_states)
                        outputs = self.sampler(logits, batch_meta_info=batch_meta_info)
                    else:
                        outputs = hidden_states
                stream.synchronize()

        sync_layers[-1]()

        return outputs
"""
        return line

    def get_norm_line(self):
        if self.norm_type == 'rms':
            norm_line = f"RMSNorm(config.hidden_size, eps=config.{self.eps_name})"
        elif self.norm_type == 'layer':
            norm_line = f"LayerNorm(config.hidden_size, eps=config.{self.eps_name})"
        else:
            raise ValueError(f'unknown norm_type:{self.norm_type}')
        return norm_line

    def get_act_line(self):
        if self.act_type == 'silu_and_mul':
            line = 'silu_and_mul'
        elif self.act_type == 'gelu':
            line = 'F.gelu'
        else:
            raise ValueError(f'unknown act:{self.act_type}')
        return line

    def generate(self, filename):
        lines = []
        lines.append(self.generate_import())
        lines.append(self.generate_mlp())
        lines.append(self.generate_moe())
        lines.append(self.generate_attn())
        lines.append(self.generate_layer())
        lines.append(self.generate_model())
        lines.append(self.generate_facade())
        line = '\n'.join(lines)

        with open(filename, 'w+') as f:
            f.write(line)




# NOTE: file with `_ignore` will not be recorded by git
postfix = '_ignore.py'   # '_ignore.py' or '.py'

# llama
scaffold = Scaffold(model_name='Llama',
                    emb_name='embed_tokens',
                    head_name='lm_head',
                    attn_name='self_attn',
                    mlp_name='mlp',
                    attn_norm_name='input_layernorm',
                    mlp_norm_name='post_attention_layernorm',
                    final_norm_name='norm',
                    norm_type='rms',
                    qkv_proj_name='q_proj,k_proj,v_proj',
                    o_proj_name='o_proj',
                    gate_up_proj_name='gate_proj,up_proj',
                    down_proj_name='down_proj',
                    act_type='silu_and_mul',
                    eps_name='rms_norm_eps',
                    num_layers_name='num_hidden_layers',
                    use_bias=(
                    'config.qkv_bias if hasattr(config, "qkv_bias") else config.attention_bias',
                    'config.o_bias if hasattr(config, "o_bias") else config.attention_bias',
                    'config.mlp_bias', 'config.mlp_bias'),
                    )
scaffold.generate(f'modeling_llama{postfix}')

# # # bailing
scaffold = Scaffold(model_name='Bailing',
                    emb_name='word_embeddings',
                    head_name='lm_head',
                    attn_name='attention',
                    mlp_name='mlp',
                    attn_norm_name='input_layernorm',
                    mlp_norm_name='post_attention_layernorm',
                    final_norm_name='norm',
                    norm_type='rms',
                    qkv_proj_name='query_key_value',
                    o_proj_name='dense',
                    gate_up_proj_name='gate_proj,up_proj',
                    down_proj_name='down_proj',
                    act_type='silu_and_mul',
                    eps_name='rms_norm_eps',
                    num_layers_name='num_layers',
                    use_bias=(
                    'config.use_qkv_bias', 'config.use_bias', 'config.use_bias',
                    'config.use_bias'),
                    )
scaffold.generate(f'modeling_bailing{postfix}')

# # qwen
scaffold = Scaffold(model_name='Qwen2',
                    emb_name='embed_tokens',
                    head_name='lm_head',
                    attn_name='self_attn',
                    mlp_name='mlp',
                    attn_norm_name='input_layernorm',
                    mlp_norm_name='post_attention_layernorm',
                    final_norm_name='norm',
                    norm_type='rms',
                    qkv_proj_name='q_proj,k_proj,v_proj',
                    o_proj_name='o_proj',
                    gate_up_proj_name='gate_proj,up_proj',
                    down_proj_name='down_proj',
                    act_type='silu_and_mul',
                    eps_name='rms_norm_eps',
                    num_layers_name='num_hidden_layers',
                    use_bias=(True, False, False, False),
                    )

scaffold.generate(f'modeling_qwen2{postfix}')

# deepseek_moe
scaffold = Scaffold(model_name='Deepseek',
                    emb_name='embed_tokens',
                    head_name='lm_head',
                    attn_name='self_attn',
                    mlp_name='mlp',
                    attn_norm_name='input_layernorm',
                    mlp_norm_name='post_attention_layernorm',
                    final_norm_name='norm',
                    norm_type='rms',
                    qkv_proj_name='q_proj,k_proj,v_proj',
                    o_proj_name='o_proj',
                    gate_up_proj_name='gate_proj,up_proj',
                    down_proj_name='down_proj',
                    act_type='silu_and_mul',
                    eps_name='rms_norm_eps',
                    num_layers_name='num_hidden_layers',
                    use_bias=(False, False, False, False),
                    moe=True,
                    num_experts_name='n_routed_experts',
                    num_shared_experts_name='n_shared_experts'
                    )
scaffold.generate(f'modeling_deepseek{postfix}')

# # bailing_moe
scaffold = Scaffold(model_name='BailingMoe',
                    emb_name='word_embeddings',
                    head_name='lm_head',
                    attn_name='attention',
                    mlp_name='mlp',
                    attn_norm_name='input_layernorm',
                    mlp_norm_name='post_attention_layernorm',
                    final_norm_name='norm',
                    norm_type='rms',
                    qkv_proj_name='query_key_value',
                    o_proj_name='dense',
                    gate_up_proj_name='gate_proj,up_proj',
                    down_proj_name='down_proj',
                    act_type='silu_and_mul',
                    eps_name='rms_norm_eps',
                    num_layers_name='num_hidden_layers',
                    use_bias=(
                    'config.use_qkv_bias', 'config.use_bias', False, False),
                    moe=True,
                    num_experts_name='num_experts',
                    num_shared_experts_name='num_shared_experts'
                    )
scaffold.generate(f'modeling_bailing_moe{postfix}')
