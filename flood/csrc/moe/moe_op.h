// The file is copied from https://github.com/vllm-project/vllm/blob/main/csrc/moe/moe_ops.h


#include <torch/extension.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_align_block_size(torch::Tensor& topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor& sorted_token_ids,
                          torch::Tensor& experts_ids,
                          torch::Tensor& num_tokens_post_pad);

void moe_sum(torch::Tensor& input,
             torch::Tensor& out);