// Copyright (c) Ant Financial Service Group and its affiliates.

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "layernorm/rmsnorm.h"
#include "activation/activation_kernels.h"
#include "rope/rope.h"
#include "cache/cache.h"
#include "moe/moe_op.h"
#include "quantize/fp8_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rmsnorm", &rmsnorm, "LayerNorm kernel");
  m.def("silu_and_mul", &silu_and_mul, "silu_and_mul kernel");
  m.def("update_cache", &update_cache, "update cache");
  m.def("update_fusion_cache", &update_fusion_cache, "update fusion cache");
  m.def("apply_rope_inplace", &apply_rope_inplace, "rope replace");
  m.def("apply_llama31_rope_inplace", &apply_llama31_rope_inplace, "llama31 rope inplace");
  m.def("apply_yarn_rope_inplace", &apply_yarn_rope_inplace, "apply yarn rope inplace");
  m.def("topk_softmax", &topk_softmax, "topk_softmax");
  m.def("moe_align_block_size", &moe_align_block_size, "moe align block size");
  m.def("moe_sum", &moe_sum, "moe sum");

  m.def("quant_to_fp8_and_update_cache", &quant_to_fp8_and_update_cache, "quant_to_fp8_and_update_cache");
  m.def("dynamic_per_token_scaled_fp8_quant", &dynamic_per_token_scaled_fp8_quant, "dynamic_per_token_scaled_fp8_quant");
  m.def("static_scaled_fp8_quant", &static_scaled_fp8_quant, "static_scaled_fp8_quant");
}