// Copyright (c) Ant Financial Service Group and its affiliates.

#pragma once
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
// #include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(pytorch_dtype, flood_type, ...)         \
  [&]() {                                                                            \
    switch (pytorch_dtype)                                                           \
    {                                                                                \
    case at::ScalarType::Half:                                                       \
    {                                                                                \
      using flood_type = half;                                                       \
      return __VA_ARGS__();                                                          \
    }                                                                                \
    case at::ScalarType::BFloat16:                                                   \
    {                                                                                \
      using flood_type = __nv_bfloat16;                                              \
      return __VA_ARGS__();                                                          \
    }                                                                                \
    default:                                                                         \
      AT_ERROR(#flood_type, " not implemented for '", toString(pytorch_dtype), "'"); \
    }                                                                                \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_FLOOD_INT_TYPE(pytorch_dtype, flood_type, ...) \
  [&]() {                                                                        \
    switch (pytorch_dtype)                                                       \
    {                                                                            \
    case at::ScalarType::Int:                                                    \
    {                                                                            \
      using flood_type = int;                                                    \
      return __VA_ARGS__();                                                      \
    }                                                                            \
    case at::ScalarType::Long:                                                   \
    {                                                                            \
      using flood_type = int64_t;                                                \
      return __VA_ARGS__();                                                      \
    }                                                                            \
    default:                                                                     \
      using flood_type = int;                                                    \
      return __VA_ARGS__();                                                      \
    }                                                                            \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_FLOOD_INT_TYPE(pytorch_dtype, flood_type, ...) \
  [&]() {                                                                        \
    switch (pytorch_dtype)                                                       \
    {                                                                            \
    case at::ScalarType::Int:                                                    \
    {                                                                            \
      using flood_type = int;                                                    \
      return __VA_ARGS__();                                                      \
    }                                                                            \
    case at::ScalarType::Long:                                                   \
    {                                                                            \
      using flood_type = int64_t;                                                \
      return __VA_ARGS__();                                                      \
    }                                                                            \
    default:                                                                     \
      using flood_type = int;                                                    \
      return __VA_ARGS__();                                                      \
    }                                                                            \
  }()

#define HEADDIM_SWITCH(head_dim, HEAD_DIM, ...) \
  [&] {                                         \
    if (head_dim == 64)                         \
    {                                           \
      constexpr uint32_t HEAD_DIM = 64;         \
      return __VA_ARGS__();                     \
    }                                           \
    else if (head_dim == 96)                    \
    {                                           \
      constexpr uint32_t HEAD_DIM = 96;         \
      return __VA_ARGS__();                     \
    }                                           \
    else if (head_dim == 128)                   \
    {                                           \
      constexpr uint32_t HEAD_DIM = 128;        \
      return __VA_ARGS__();                     \
    }                                           \
    else                                        \
    {                                           \
      constexpr uint32_t HEAD_DIM = 256;        \
      return __VA_ARGS__();                     \
    }                                           \
  }()

// static inline __device__ float to_float(half src)
// {
//     return __half2float(src);
// }

// static inline __device__ float to_float(__nv_bfloat16 src)
// {
//     return  __bfloat162float(src);
// }

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT flood_cast(T_IN val) { return val; }
template <>
__device__ inline __nv_bfloat16 flood_cast<__nv_bfloat16, float>(float val) { return __float2bfloat16(val); }
template <>
__device__ inline half flood_cast<half, float>(float val) { return __float2half(val); }
template <>
__device__ inline float flood_cast<float, __nv_bfloat16>(__nv_bfloat16 val) { return __bfloat162float(val); }
template <>
__device__ inline float flood_cast<float, half>(half val) { return __half2float(val); }
