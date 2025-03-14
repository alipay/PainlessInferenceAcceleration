// Copyright (c) Ant Financial Service Group and its affiliates.

#include <torch/extension.h>

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);