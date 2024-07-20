# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import time
import sys


model_type = 'antglm_distill'
if model_type == 'antglm_default':
    # 非墨 default
    model_dir = '/mntnlp/feimo/antglm/model/ant_glm_sft_ref_crawler_v2.pt'
elif model_type == 'antglm_service':
    # 恬莫 service
    model_dir = '/mntnlp/tianmo/glm/271098/tool_learning_mix_3_mlp6/last.ckpt'
elif model_type == 'antglm_gov':
    model_dir = '/mntnlp/yumu/glm/258258/sft_cot_zhengwu/cot_v2.pt'
elif model_type == 'antglm_distill':
    model_dir = '/mntnlp/feimo/antglm/model/glm10b_sft_single_doc_new.pt'
print('start loading ckpt')
ts = time.time()
state_dict = torch.load(model_dir,map_location=torch.device('cuda:0'))['state_dict']

print(f'load ckpt in {round(time.time()-ts,1)}s')

state_dict = {k:v.bfloat16() for k,v in state_dict.items()}

print(f'convert dtype in {round(time.time()-ts,1)}s')


p = 'model.glm'
rename_state_dict = {}
rename_state_dict['wte.weight'] = state_dict[f'{p}.word_embeddings.weight']
rename_state_dict['wpe.weight'] = state_dict[f'{p}.transformer.position_embeddings.weight']
rename_state_dict['wbe.weight'] = state_dict[f'{p}.transformer.block_position_embeddings.weight']

rename_state_dict['ln_f.weight'] = state_dict[f'{p}.transformer.final_layernorm.weight']
rename_state_dict['ln_f.bias'] = state_dict[f'{p}.transformer.final_layernorm.bias']
for i in range(48):
    rename_state_dict[f'h.{i}.ln_1.weight'] = state_dict[f'{p}.transformer.layers.{i}.input_layernorm.weight']
    rename_state_dict[f'h.{i}.ln_1.bias'] = state_dict[f'{p}.transformer.layers.{i}.input_layernorm.bias']
    rename_state_dict[f'h.{i}.attn.c_attn.weight'] = state_dict[f'{p}.transformer.layers.{i}.attention.query_key_value.weight']
    rename_state_dict[f'h.{i}.attn.c_attn.bias'] = state_dict[f'{p}.transformer.layers.{i}.attention.query_key_value.bias']
    rename_state_dict[f'h.{i}.attn.c_proj.weight'] = state_dict[f'{p}.transformer.layers.{i}.attention.dense.weight']
    rename_state_dict[f'h.{i}.attn.c_proj.bias'] = state_dict[f'{p}.transformer.layers.{i}.attention.dense.bias']
    rename_state_dict[f'h.{i}.ln_2.weight'] = state_dict[f'{p}.transformer.layers.{i}.post_attention_layernorm.weight']
    rename_state_dict[f'h.{i}.ln_2.bias'] = state_dict[f'{p}.transformer.layers.{i}.post_attention_layernorm.bias']
    rename_state_dict[f'h.{i}.mlp.c_fc.weight'] = state_dict[f'{p}.transformer.layers.{i}.mlp.dense_h_to_4h.weight']
    rename_state_dict[f'h.{i}.mlp.c_fc.bias'] = state_dict[f'{p}.transformer.layers.{i}.mlp.dense_h_to_4h.bias']
    rename_state_dict[f'h.{i}.mlp.c_proj.weight'] = state_dict[f'{p}.transformer.layers.{i}.mlp.dense_4h_to_h.weight']
    rename_state_dict[f'h.{i}.mlp.c_proj.bias'] = state_dict[f'{p}.transformer.layers.{i}.mlp.dense_4h_to_h.bias']
del state_dict
# rename_state_dict = {k:v.bfloat16() for k,v in rename_state_dict.items()}
for k,v in rename_state_dict.items():
    print(k,v.shape)
model_name = f'/mntnlp/nanxiao/{model_type}_prefetch/pytorch_model.bin'
torch.save(rename_state_dict, model_name)
print(f'model is saved to {model_name} in {round(time.time()-ts,1)}s')

