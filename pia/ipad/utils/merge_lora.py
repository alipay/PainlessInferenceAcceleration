# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import time
import torch
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


model_path = '/ossfs/workspace/distill/blip/nas/data/workspace/wupeiwen.wpw/llm_models/glm5b_sft'
lora_path = '/ossfs/workspace/distill/blip/nas/data/workspace/wupeiwen.wpw/experiments/geo_multi_task/train_peft_ds_seq2seq_glm5b.sh/20230718-152844'

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

# peft_config = LoraConfig.from_pretrained(path2sftconfig)
model = PeftModel.from_pretrained(model, lora_path)
model.merge_and_unload()

model = model.half().cuda()
model.eval()

state_dict = model.state_dict()

ts = time.time()
p = 'base_model.model.glm'
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
model_name = f'/mntnlp/nanxiao/poi/pytorch_model.bin'
torch.save(rename_state_dict, model_name)
print(f'model is saved to {model_name} in {round(time.time()-ts,1)}s')