// Copyright (c) Ant Financial Service Group and its affiliates.

#include <torch/extension.h>

void update_cache(torch::Tensor& k_out,  torch::Tensor& v_out, torch::Tensor& key_states, torch::Tensor& value_states,  
                  torch::Tensor& indices, int tok, int dim, int stride);

void update_fusion_cache(torch::Tensor& kv_out,  torch::Tensor& kv_states, 
                  torch::Tensor& indices, int tok, int dim, int stride);
