# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from pia.ipad.models.opt.opt_trainer import OptDistillWorker,DistillPipe

algo = 'kl'
# token_dir = '/mntnlp/common_base_model/opt_6b7'
# # model_dir = '/mntnlp/common_base_model/opt_6b7'
# model_dir = '/mnt_alipayshnas/workspace/nanxiao/opt_kl_sft_7'
# sample_dir = '/mntnlp/nanxiao/opt/dolly_15k.jsonl'
# logit_dir = '/mntnlp/nanxiao/opt/opt_logits_15k.npz'
# save_dir = f'/mnt_alipayshnas/workspace/nanxiao/opt_{algo}_madd'
# load_dir = '/mnt_alipayshnas/workspace/nanxiao/opt_kl_madd_dim_0'
# log_dir = './logs/opt'

token_dir ='/Users/yaozhao/dataset/models/opt-125m'
model_dir = '/Users/yaozhao/dataset/models/opt-125m'
sample_dir = '/Users/yaozhao/dataset/dataset/dolly_15k.jsonl'
logit_dir = '/Users/yaozhao/dataset/models/llama/llama2_7b_logits_15k.npz'
save_dir = f'/Users/yaozhao/dataset/models/llama/llama2_7b_{algo}_madd'
load_dir = '/Users/yaozhao/dataset/models/llama/llama2_7b_kl_madd_sft_7'
log_dir = '.opt'


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
train_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
pred_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
sample_count = 16
bs = 4
mbs = 1
pbs = 4
train_stages = 2
save_stages = []
itv = 2
optim = 'AdamW'
lr = 0.001
lrs = 'cos'
layer_loss_coefs = {"kl": 1.0}
if algo == 'kl':
    model_loss_coefs = {"kl": 1.0}
elif algo == 'pair':
    model_loss_coefs = {"kl": 1.0, 'pair': 0.1}
elif algo == 'ppo':
    model_loss_coefs = {"kl": 1.0, "ppo": 0.1}
else:
    raise ValueError(f'unknown algo:{algo}')


use_load = False
use_cache = False
use_aux_cache = False
use_mlp = False
use_attn = False
use_dim = True
use_depth = True
use_mlp_reparam = False
use_attn_reparam = False
use_dim_reparam = False
use_freeze = False
use_refit = False
use_final = False

worker = OptDistillWorker(sample_dir=sample_dir,
                              logit_dir=logit_dir,
                              log_dir=log_dir,
                              max_input_len=8,
                              max_gen_len=8,
                              train_dtype=train_dtype,
                              pred_dtype=pred_dtype,
                              device=device,
                              log_steps=1,
                              eval_steps=100,
                              eval_count=1)

pipe = DistillPipe(worker=worker,
                   sample_count=sample_count,
                   bs=bs,
                   mbs=mbs,
                   pbs=pbs,
                   itv=itv,
                   train_stages=train_stages,
                   optim=optim,
                   lr=lr,
                   lrs=lrs,
                   layer_loss_coefs=layer_loss_coefs,
                   model_loss_coefs=model_loss_coefs,
                   save_dir=save_dir,
                   conf_dir=model_dir,
                   token_dir=token_dir,
                   save_stages=save_stages
                   )
#
# pipe.finetune(model_dir=model_dir,
#               use_split=True,
#               split_rate_or_count=0.5,
#               use_shrink=False,
#               shrink_rate_or_count=None,
#               lr=0.01,
#               train_stages=8,
#               save_stages=range(10))
# model_dir = '/mnt_alipayshnas/workspace/nanxiao/opt_kl_madd_sft_3'
# NOTE: patch
# exit(0)

pipe.initialize(use_cache=use_cache,
                 use_aux_cache=use_aux_cache,
                 model_dir=model_dir,
                 use_chat=False,
                 use_acc=True,
                 use_split=True,
                 split_rate_or_count=0.5,
                 use_shrink=False,
                 shrink_rate_or_count=None,
                 eval_count=20)

# pipe.load_ckpt(use_load=use_load,
#                load_dir=load_dir,
#                eval_count=20)

worker.star_log(f'exp:15k loss:{algo}')
pipe.mlp_prune(use_mlp=use_mlp,    segs=[i/16 for i in range(train_stages+1)], layer_steps=4*pipe.steps, model_steps=4*pipe.steps, save_stages=range(0))
pipe.attn_prune(use_attn=use_attn, segs=[i/32 for i in range(train_stages+1)], layer_steps=4*pipe.steps, model_steps=4*pipe.steps, save_stages=range(0))
pipe.dim_prune(use_dim=use_dim,    segs=[i/32 for i in range(train_stages+1)], layer_steps=4*pipe.steps, model_steps=4*pipe.steps, save_stages=range(0))
pipe.depth_prune(use_depth=use_depth, segs=range(2,0,-1),  layer_steps=0, model_steps=4*pipe.steps, save_stages=range(31,0,-1))
pipe.mlp_reparam(use_mlp_reparam=use_mlp_reparam)
pipe.attn_reparam(use_attn_reparam=use_attn_reparam)
pipe.dim_reparam(use_dim_reparam=use_dim_reparam)
pipe.freeze_param(use_freeze=use_freeze)
pipe.refit_param(use_refit=True, train_stages=8, model_steps=1000, save_stages=range(8))
pipe.final_export(use_final=use_final)

print('done!')
