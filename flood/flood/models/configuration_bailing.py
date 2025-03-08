#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

""" Bailing model configuration """

from transformers.configuration_utils import PretrainedConfig


class BailingConfig(PretrainedConfig):
    model_type = "bailing"

    def __init__(
            self,
            vocab_size=30592,
            hidden_size=1024,
            intermediate_size=None,  # todo 默认值
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=0,
            hidden_act="silu",
            use_qkv_bias=False,
            use_bias=True,
            rms_norm_eps=1e-06,
            norm_softmax=False,
            norm_head=False,
            tie_word_embeddings=True,
            embedding_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            output_dropout_prob=0.1,
            initializer_range=0.02,
            max_position_embeddings=16384,
            rope_theta=10000.0,
            use_cache=True,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            **kwargs,
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_softmax = norm_softmax
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps

        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.initializer_range = initializer_range

        self.max_position_embeddings = max_position_embeddings

        self.rope_theta = rope_theta
        self.use_cache = use_cache

        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
