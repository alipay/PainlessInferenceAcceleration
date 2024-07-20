# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from pia.ipad.models.antgmm.antgmm_trainer import AntgmmDistillWorker, DistillPipe


model_dir = '/mntnlp/nanxiao/antgmm'
token_dir = '/mntnlp/nanxiao/antgmm'

sample_dir = '/mntnlp/nanxiao/antgmm/qas_16k.txt'
emb_dir = '/mntnlp/nanxiao/blip/embs_16k.bin'
logit_dir = '/mntnlp/nanxiao/blip/logits_16k.npz'

aux_sample_dir = '/mntnlp/nanxiao/antgmm/test_qas_3k.txt'
aux_emb_dir = '/mntnlp/nanxiao/antgmm/test_embs_3k.bin'
aux_logit_dir = '/mntnlp/nanxiao/antgmm/test_logits_3k.npz'

save_dir = '/mnt_alipayshnas/workspace/nanxiao/antgmm_exp_v7_p20_only_mlp'
# load_dir = '/mnt_alipayshnas/workspace/nanxiao/antgmm_exp_v7_depth_iterstep_refit'
load_dir = None
log_dir = 'antgmm'

algo = 'kl'
device=torch.device('cuda:0')
sample_count = None
bs = 1024
mbs = 8
pbs = 16
itv = 1
stages = 8
optim = 'AdamW'
lr = 0.003
lrs = 'cos'
layer_loss_coefs = {"dis": 1.0}
model_loss_coefs = {"dis": 1.0}
save_stages = []

use_sft = False
use_load = False
use_wait = False
use_cache = True

use_depth = False
use_mlp = True
use_attn = False
use_dim = False

use_mlp_reparam = False
use_attn_reparam = False
use_dim_reparam = False

use_freeze = False
use_refit = False
use_final = False

worker = AntgmmDistillWorker(sample_dir=sample_dir,
                             logit_dir=logit_dir,
                             emb_dir=emb_dir,
                             aux_sample_dir=aux_sample_dir,
                             aux_logit_dir=aux_logit_dir,
                             aux_emb_dir=aux_emb_dir,
                             log_dir=log_dir,
                             max_input_len=256,
                             max_gen_len=32,
                             train_dtype=torch.bfloat16,
                             pred_dtype=torch.float16,
                             device=device,
                             log_steps=1,
                             eval_steps=5,
                             eval_count=300)

pipe = DistillPipe(worker=worker,
                   sample_count=sample_count,
                   bs=bs,
                   mbs=mbs,
                   pbs=pbs,
                   itv=itv,
                   train_stages=stages,
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

pipe.initialize(
                 use_cache=use_cache,
                 model_dir=model_dir,
                 use_chat=False,
                 use_acc=True,
                 use_split=False,
                 split_rate_or_count=None,
                 use_shrink=False,
                 shrink_rate_or_count=None,
                 eval_count=2500)

worker.star_log('exp:shrink dataset, rate=0.2, only mlp')
pipe.load_ckpt(use_load=use_load, load_dir=load_dir, eval_count=500)
pipe.depth_prune(use_depth=use_depth, segs=[36,24,12,8],  layer_steps=0, model_steps=4*pipe.steps, save_stages=[8])
pipe.mlp_prune(use_mlp=use_mlp,    segs=(0,8/9), stages=8, layer_steps=2*pipe.steps, model_steps=4*pipe.steps, save_stages=[7])
pipe.attn_prune(use_attn=use_attn, segs=(0,1/4), stages=8, layer_steps=2*pipe.steps, model_steps=4*pipe.steps, save_stages=[7])
pipe.dim_prune(use_dim=use_dim,    segs=(0,1/4), stages=8, layer_steps=4*pipe.steps, model_steps=8*pipe.steps, save_stages=[7])
pipe.mlp_reparam(use_mlp_reparam=use_mlp_reparam)
pipe.attn_reparam(use_attn_reparam=use_attn_reparam)
pipe.dim_reparam(use_dim_reparam=use_dim_reparam)
pipe.freeze_param(use_freeze=use_freeze)
pipe.refit_param(use_refit=True, model_steps=1000000)
pipe.final_export(use_final=use_final)

print('done!')
