# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


from __future__ import print_function

import time
import os
import json

from rouge_score import rouge_scorer

import torch

from transformers import LlamaForCausalLM, LlamaTokenizer
from ipad.common.distill_worker import DistillWorker
from ipad.common.sparse_module import LlamaSparseDim,LlamaSparseAttn,LlamaSparseMLP, \
    SparseRMSNorm

torch.manual_seed(7)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class LlamaDistillWorker(DistillWorker):
    mlp_name = 'mlp'
    attn_name = 'self_attn'
    ln1_name = 'input_layernorm'
    ln2_name = 'post_attention_layernorm'

    def __init__(self,
                 sample_dir=None, logit_dir=None, emb_dir=None, emb_idx=1,
                 aux_sample_dir=None, aux_logit_dir=None, aux_emb_dir=None,
                 log_dir=None, log_steps=1, eval_steps=5, eval_count=100,
                 max_input_len=256, max_gen_len=64,
                 train_dtype=torch.bfloat16,
                 pred_dtype=torch.float16,
                 device=None):
        super(LlamaDistillWorker, self).__init__(log_dir=log_dir)
        self.sample_dir = sample_dir
        self.logit_dir = logit_dir
        self.emb_dir = emb_dir
        self.emb_idx = emb_idx
        self.aux_sample_dir = aux_sample_dir
        self.aux_logit_dir = aux_logit_dir
        self.aux_emb_dir = aux_emb_dir
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.eval_count = eval_count
        self.max_input_len = max_input_len
        self.max_gen_len = max_gen_len
        self.train_dtype = train_dtype
        self.pred_dtype = pred_dtype
        self.device = device

    def _model_from_pretrained(self, model_dir, dtype=torch.bfloat16):
        model = LlamaForCausalLM.from_pretrained(model_dir,
                                                 cache_dir='/',
                                                 torch_dtype=dtype,
                                                 device_map="auto")
        return model

    def _tokenizer_from_pretrained(self, token_dir):
        tokenizer = LlamaTokenizer.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_sparse_norm(self, layer=None, mask=None):
        return SparseRMSNorm(layer=layer, mask=mask)

    def _get_sparse_mlp(self, layer=None, layer_index=0):
        return LlamaSparseMLP(layer=layer, layer_index=layer_index)

    def _get_sparse_attn(self, layer=None, layer_index=0):
        return LlamaSparseAttn(layer=layer, layer_idx=layer_index)

    def _get_sparse_dim(self, layer=None, layer_index=0, mask=None):
        return LlamaSparseDim(layer=layer, layer_idx=layer_index, mask=mask)

    def get_layers(self):
        return self.model.model.layers

    def set_layers(self, layers):
        self.model.model.layers = layers

    def get_final_norm(self):
        return self.model.model.norm

    def set_final_norm(self, norm):
        self.model.model.norm = norm

    def get_transformer(self):
        return self.model.model

    def set_transformer(self, transformer):
        self.model.model = transformer

    def get_head(self):
        return self.model.lm_head

    def set_head(self, head):
        self.model.lm_head = head

    def get_wte(self):
        return self.model.model.embed_tokens

    def set_wte(self, wte):
        self.model.model.wte = wte

    def get_wpe(self):
        return None

    def set_wpe(self, wpe):
        pass

    @property
    def n_layer(self):
        return len(self.model.model.layers)

    def forward(self, input_ids, position_ids=None, attention_mask=None, embeddings=None):
        outputs = self.get_transformer().forward(input_ids,
                                                 attention_mask=attention_mask)
        return outputs[0]

    def est_params(self, head=False, tied=True):
        n_params = 0
        depth = self.n_layer
        dim = self.dim - self.mask_counts.get('dim', 0)
        attn_dim = self.attn_dim - self.mask_counts.get('attn', 0)
        mlp_dim = self.mlp_dim - self.mask_counts.get('mlp', 0)
        if head:
            n_params += self.n_voc * dim * (1 if tied else 2)
        n_params += (3 * dim * mlp_dim + 2*mlp_dim + dim) * depth
        n_params += ( 4 * dim * attn_dim + 3 * attn_dim + dim) * depth
        n_params += (depth * 2 + 1) * dim  # layernorm
        return n_params

    def save_conf(self, save_dir=None, prefix='LlamaConfig'):
        ts = time.time()
        conf = self.model.config

        conf.num_hidden_layers = self.n_layer

        conf.hidden_size = self.get_head().weight.size(1)
        mlp = getattr(self.get_layers()[0], self.mlp_name)
        l1 = mlp.up_proj
        conf.intermediate_size = l1.weight.size(0)

        attn = getattr(self.get_layers()[0], self.attn_name)
        l1 = attn.q_proj
        conf.attention_size = l1.weight.size(0)  # TODO
        conf.tie_word_embeddings = False

        conf_str = str(conf)[len(prefix) + 1:]
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + '/config.json', 'w') as f:
            f.write(conf_str)

        self.log(f'save conf:{save_dir.split(",")[-1]} in {round(time.time() - ts, 3)}s')

    def calc_acc(self, count=100, batch_size=1, ds='test', max_log_count=0, filename=None, info=''):
        assert ds in ('test', 'train', 'aux')
        count = 1000000 if count is None else count

        ts = time.time()
        if ds == 'test':
            queries = self.test_queries
            answers = self.test_answers
            embeddings = self.test_embeddings
        elif ds == 'train':
            queries = self.queries
            answers = self.answers
            embeddings = self.embeddings
        elif ds == 'aux':
            queries = self.aux_queries
            answers = self.aux_answers
            embeddings = self.aux_embeddings

        if embeddings is not None:
            embeddings = embeddings[:count]

        checkpoint = self.get_transformer().gradient_checkpointing
        self.get_transformer().gradient_checkpointing = False
        rs = self.batch_chat(queries[:count],
                             embeddings=embeddings,
                             batch_size=batch_size,
                             max_length=self.max_gen_len,
                             max_log_count=max_log_count,
                             log_query=False,
                             log_truth=False,
                             log_answer=True,
                             log_space=False,
                             emit=True)
        te = time.time()
        
        self.get_transformer().gradient_checkpointing = checkpoint

        if filename is not None:
            jsons = []
            for i, q in enumerate(queries[:count]):
                r = rs[i]
                a = answers[i]
                jsons.append(json.dumps({'prompt':q, 'response': r, 'truth': a}))
            with open(filename, 'w') as f:
                f.write('\n'.join(jsons))

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        te = time.time()
        n_p = 0
        n = 0
        n_ps = [0] * ((len(rs) - 1) // 100 + 1)
        score_details = [[],[]]
        for i, ans in enumerate(rs):
            pred = ans.replace('<s>', '').replace('</s>', '').strip()
            true = answers[i].replace('<s>', '').replace('</s>', '').strip()
            n += 1
            score = scorer.score(prediction=pred, target=true)["rougeL"].fmeasure
            n_p += score
            n_ps[i // 100] += score
            if 'paired with an input' in queries[i]:
                score_details[0].append(score)
            else:
                score_details[1].append(score)
        acc = n_p / n
        details = [round(x / 100, 3) for x in n_ps]
        score_details = [round(sum(x)/max(len(x),1), 3) for x in score_details]
        size = self.est_params()/1e9
        self.log(f'\nrouge:{acc:.3f} sample:{n} details:{details}/{score_details} size:{size:.3f} elapse:{te - ts:.3f} {info}\n')
        return acc

