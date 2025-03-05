// Copyright (c) Ant Financial Service Group and its affiliates.

#include <torch/extension.h>

void rmsnorm(torch::Tensor& _input, torch::Tensor& _gamma, torch::Tensor& _out, float eps);