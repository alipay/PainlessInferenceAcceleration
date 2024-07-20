# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from pia.ipad.common.distill_worker import DistillWorker,DistillPipe
from pia.ipad.models.glm.glm_trainer import GlmDistillWorker

#
# ds = 'searchx_ext'
# if ds == 'gui':
#     token_dir = '/mntnlp/nanxiao/guix'
#     model_dir = '/mntnlp/nanxiao/guix'
#
#     sample_dir = '/mntnlp/nanxiao/guix/qas_2k.txt'
#     logit_dir = '/mntnlp/nanxiao/guix/logits_2k.npz'
#
#     aux_sample_dir = '/mntnlp/nanxiao/guix/test_qas_100.txt'
#     aux_logit_dir = '/mntnlp/nanxiao/guix/test_logits_100.npz'
#
#     save_dir = '/mntnlp/nanxiao/searchs_exp_v6'
#     load_dir = None
#     log_dir = 'guix'
#
#     max_input_len = 4096 - 32
#     max_gen_len = 32
#
# elif ds == 'searchx_ext': # searchx.extract
#
#     token_dir = '/mntnlp/nanxiao/searchx_ext'
#     model_dir = '/mntnlp/nanxiao/searchx_ext'
#
#     sample_dir = '/mntnlp/nanxiao/searchx_ext/train_64k_short.jsonl'
#     logit_dir = '/mntnlp/nanxiao/searchx_ext/train_logits_64k_short.npz'
#
#     aux_sample_dir = '/mntnlp/nanxiao/searchx_ext/test_400_short.jsonl'
#     aux_logit_dir = '/mntnlp/nanxiao/searchx_ext/test_logits_400_short.npz'
#
#     save_dir = '/mnt_alipayshnas/workspace/nanxiao/searchx_ext_v6'
#     load_dir = None
#     log_dir = 'searchx_ext'
#
#     max_input_len = 2048 - 256
#     max_gen_len = 256
#
# elif ds == 'searchx_rew': # searchx.rewrite
#
#     token_dir = '/mntnlp/luohe/exp/searchx/multidoc/antglm_100k_1118'
#     model_dir = '/mntnlp/luohe/exp/searchx/multidoc/antglm_100k_1118'
#
#     sample_dir = '/mntnlp/nanxiao/searchx/train_10k_trunc.jsonl'
#     logit_dir = '/mntnlp/nanxiao/searchx/train_logits_10k_trunc.npz'
#
#     aux_sample_dir = '/mntnlp/nanxiao/searchx/test_2k_short.jsonl'
#     aux_logit_dir = '/mntnlp/nanxiao/searchx/test_logits_2k_short.npz'
#
#     save_dir = '/mnt_alipayshnas/workspace/nanxiao/searchx_v8_trunc'
#     load_dir = '/mnt_alipayshnas/workspace/nanxiao/searchx_v8_trunc'
#     log_dir = 'searchx_rew'
#
#     max_input_len = 2048 - 256
#     max_gen_len = 256
#
# else:
#     raise ValueError(f'unknown ds:{ds}')
#
algo = 'kl'
token_dir ='/Users/yaozhao/dataset/models/antglm'
model_dir = '/Users/yaozhao/dataset/models/antglm'
sample_dir = '/Users/yaozhao/dataset/dataset/dolly_15k.jsonl'
logit_dir = '/Users/yaozhao/dataset/models/llama/llama2_7b_logits_15k.npz'
save_dir = f'/Users/yaozhao/dataset/models/llama/llama2_7b_{algo}_madd'
load_dir = '/Users/yaozhao/dataset/models/llama/llama2_7b_kl_madd_sft_7'
log_dir = './logs/llama'


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
train_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
pred_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

sample_count = 8
bs = 4
mbs = 1
pbs = 4
stages = 2
itv = 8
optim = 'SGD'
lr = 0.001
lrs = 'cos'
layer_loss_coefs = {"kl": 1.0}
model_loss_coefs = {"kl": 1.0}
save_stages = []

use_sft = False
use_load = False
use_wait = False
use_cache = True
use_aux_cache = True
use_depth = True
use_mlp = True
use_attn = False
use_dim = False
use_mlp_reparam = False
use_attn_reparam = False
use_dim_reparam = False
use_freeze = False
use_refit = False
use_final = False

worker = GlmDistillWorker(sample_dir=sample_dir,
                          logit_dir=logit_dir,
                          # aux_sample_dir=aux_sample_dir,
                          # aux_logit_dir=aux_logit_dir,
                          log_dir=log_dir,
                          max_input_len=64,
                          max_gen_len=8,
                          train_dtype=torch.bfloat16,
                          pred_dtype=torch.bfloat16,
                          device=device,
                          log_steps=1,
                          eval_steps=5,
                          eval_count=100,
                          error_count=5)

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

# pipe.finetune(model_dir=model_dir,
#               use_split=True,
#               split_rate_or_count=1000,
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
                 use_chat=True,
                 chat_count=1,
                 use_acc=True,
                 use_split=True,
                 split_rate_or_count=0.5,
                 use_shrink=False,
                 shrink_rate_or_count=None,
                 eval_count=4)


pipe.load_ckpt(use_load=use_load,
               load_dir=load_dir,
               eval_count=4)

# worker.calc_acc(100000, ds='test', batch_size=4, filename='/mntnlp/nanxiao/searchx_ext/pred_test_qas.jsonl')
# worker.calc_acc(100000, ds='train', batch_size=4, filename='/mntnlp/nanxiao/searchx/pred_train_qas.jsonl')

worker.star_log('exp:fast  ds:short, opt:sgd')
pipe.depth_prune(use_depth=use_depth, segs=range(40,22,-4), layer_steps=0, model_steps=pipe.steps, save_stages=[],checkpoint=True)
pipe.mlp_prune(use_mlp=use_mlp,    segs=(0,1/2), stages=8, layer_steps=2*pipe.steps, model_steps=4*pipe.steps, save_stages=[])
pipe.attn_prune(use_attn=use_attn, segs=(0,1/4), stages=8, layer_steps=2*pipe.steps, model_steps=4*pipe.steps, save_stages=[])
pipe.dim_prune(use_dim=use_dim,    segs=(0,1/4), stages=8, layer_steps=4*pipe.steps, model_steps=8*pipe.steps, save_stages=[])
pipe.mlp_reparam(use_mlp_reparam=use_mlp_reparam)
pipe.attn_reparam(use_attn_reparam=use_attn_reparam)
pipe.dim_reparam(use_dim_reparam=use_dim_reparam)
pipe.freeze_param(use_freeze=use_freeze)
pipe.refit_param(use_refit=True, model_steps=1000000)
pipe.final_export(use_final=use_final)

print('done!')