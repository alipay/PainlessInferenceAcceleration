// Copyright (c) Ant Financial Service Group and its affiliates.


#include <torch/extension.h>

void apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_rope, torch::Tensor k_rope,
                torch::Tensor indptr, torch::Tensor offsets, bool interleave, float rope_scale,
                float rope_theta);


void apply_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                        torch::Tensor offsets, bool interleave, float rope_scale, float rope_theta);


void apply_llama31_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                                torch::Tensor offsets, bool interleave, float rope_scale,
                                float rope_theta, float low_freq_factor, float high_freq_factor,
                                float old_context_length);

void apply_yarn_rope_inplace(torch::Tensor q, torch::Tensor k, torch::Tensor indptr,
                             torch::Tensor offsets, bool interleave, float rope_scale, float rope_theta,
                             float low, float high, float attention_factor);

