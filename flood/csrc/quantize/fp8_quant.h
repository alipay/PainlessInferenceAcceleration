// Copyright (c) Ant Financial Service Group and its affiliates.


#include <torch/extension.h>

void static_scaled_fp8_quant(torch::Tensor out, torch::Tensor const input,
                             float scale);


void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor out, torch::Tensor const input, torch::Tensor scale,
    c10::optional<torch::Tensor> const scale_ub);


void quant_to_fp8_and_update_cache(torch::Tensor& q_out, torch::Tensor& k_out,  torch::Tensor& v_out, 
                                        torch::Tensor& query_states, torch::Tensor& key_states, torch::Tensor& value_states,  
                                        torch::Tensor& indices, 
                                        int tok, int group, int kv_dim, int q_stride, int kv_stride);