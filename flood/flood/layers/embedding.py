# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch


class AutoEmbedding():

    @classmethod
    def from_pretrained(cls,
                        config,
                        vocab_size,
                        hidden_size,
                        padding_idx=None,
                        name="embed_tokens"):
        emb_dtype = config.emb_dtype if hasattr(config,
                                                "emb_dtype") else None  # TODO: parse quantization config
        model_dtype = config.torch_dtype
        if emb_dtype is None or emb_dtype in ('float16', 'bfloat16'):
            return NativeEmbedding(vocab_size, hidden_size,
                                   padding_idx=padding_idx,
                                   model_dtype=model_dtype)
        elif emb_dtype == 'float8_e4m3fn':
            return Fp8Embedding(vocab_size, hidden_size,
                                padding_idx=padding_idx,
                                model_dtype=model_dtype)
        else:
            raise ValueError(f"unknown dtype:{emb_dtype}")


class NativeEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx=None, weight=None,
                 model_dtype=torch.bfloat16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.model_dtype = model_dtype
        if weight is None:
            weight = torch.empty(vocab_size, hidden_size, dtype=model_dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, indices):
        return self.weight.data[indices]

    def retype(self, dtype=torch.float8_e4m3fn):
        if dtype == torch.float8_e4m3fn:
            scale = torch.max(self.weight.data.abs(), dim=1,
                              keepdim=True).values.to(self.model_dtype) / 448.0
            weight = (self.weight / scale).to(dtype)
            self.weight = None
            delattr(self, 'weight')
            emb = Fp8Embedding(self.vocab_size, self.hidden_size,
                               padding_idx=self.padding_idx, weight=weight,
                               scale=scale, model_dtype=self.model_dtype)
            return emb
        else:
            raise ValueError(f"unknown dtype:{dtype}")


class Fp8Embedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx=None, weight=None,
                 scale=None, model_dtype=torch.bfloat16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.model_dtype = model_dtype
        if weight is None or scale is None:
            weight = torch.empty(vocab_size, hidden_size,
                                 dtype=torch.float8_e4m3fn)
            scale = torch.empty(vocab_size, 1, dtype=model_dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.scale = scale

    def forward(self, indices):
        return self.weight.data.view(torch.int8)[indices].view(
            torch.float8_e4m3fn).to(self.model_dtype) * self.scale[indices]
