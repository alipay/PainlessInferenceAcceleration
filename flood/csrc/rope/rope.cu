/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
Adapt (Heavily) from https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/pos_enc.cuh

Modified by Chen Liang
*/

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "vec_types.h"

__host__ __device__ __forceinline__ size_t get_elem_offset_impl(size_t elem_idx, size_t head_idx,
                                                                size_t feat_idx, size_t stride_n,
                                                                size_t stride_h) {
  return elem_idx * stride_n + head_idx * stride_h + feat_idx;
}

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }


template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_yarn_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    float a_factor, const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);
  permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                  ? threadIdx.x * vec_size + rotary_dim / 2
                                  : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    float embed = float(offset) * freq[i];
    float cos, sin;
    __sincosf(embed, &sin, &cos);
    vec[i] = vec[i] * cos * a_factor +
             ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin * a_factor;
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_yarn_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    float a_factor, const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos * a_factor + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin * a_factor;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] =
          vec[i] * cos +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin;
    }
  }
  return vec;
}


/******************** apply rope ********************/

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float smooth_a, float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        DType* q_rope_ptr =
            q_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                          q_rope_stride_n, q_rope_stride_h);
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          q_vec =
              vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }
  } else {
    // apply rotary to k
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        DType* k_rope_ptr =
            k_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                          k_rope_stride_n, k_rope_stride_h);
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          k_vec =
              vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);
      }
    }
  }
}



template <typename DType, typename IdType>
void BatchQKApplyRotary(DType* q, DType* k, DType* q_rope, DType* k_rope,
                        IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                        uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                        uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n,
                        size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, 
                        size_t q_rope_stride_n, size_t q_rope_stride_h, 
                        size_t k_rope_stride_n, size_t k_rope_stride_h, bool interleave,
                        float rope_scale, float rope_theta, cudaStream_t stream = nullptr){
        
    float rope_rcp_scale = 1.0f / rope_scale;
    float rope_rcp_theta = 1.0f / rope_theta;
    float smooth_a = 0.f;
    float smooth_b = 0.f;

    constexpr uint32_t HEAD_DIM = 128;

    // constexpr uint32_t vec_size = std::max(16 / sizeof(DType), head_dim / 32);
    constexpr uint32_t vec_size = 16 / sizeof(DType);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = num_threads / bdx;
    dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
    dim3 nthrs(bdx, bdy);

    const bool INTERLEAVE = false; 
    
    BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>
    <<<nblks, nthrs, 0, stream>>>(  q,
                                    k,
                                    q_rope,
                                    k_rope,
                                    indptr,
                                    offsets,
                                    batch_size,
                                    num_qo_heads,
                                    num_kv_heads,
                                    rotary_dim,
                                    q_stride_n,
                                    q_stride_h,
                                    k_stride_n,
                                    k_stride_h,
                                    q_rope_stride_n,
                                    q_rope_stride_h,
                                    k_rope_stride_n,
                                    k_rope_stride_h,
                                    smooth_a,
                                    smooth_b,
                                    rope_rcp_scale,
                                    rope_rcp_theta);
}


void apply_rope(torch::Tensor& q, torch::Tensor& k, torch::Tensor& q_rope, torch::Tensor& k_rope, torch::Tensor& indptr,
                torch::Tensor& offsets, int64_t rotary_dim, bool interleave, float rope_scale,
                float rope_theta) {

    auto device = q.device();
    unsigned int num_qo_heads = q.size(1);
    unsigned int num_kv_heads = k.size(1);
    unsigned int head_dim = q.size(2);
    unsigned int batch_size = offsets.size(0);
    size_t q_stride_n = q.stride(0);
    size_t q_stride_h = q.stride(1);
    size_t k_stride_n = k.stride(0);
    size_t k_stride_h = k.stride(1);
    size_t q_rope_stride_n = q_rope.stride(0);
    size_t q_rope_stride_h = q_rope.stride(1);
    size_t k_rope_stride_n = k_rope.stride(0);
    size_t k_rope_stride_h = k_rope.stride(1);
    indptr = indptr.to(torch::kInt32);
    offsets = offsets.to(torch::kInt32);

    cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
    DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(q.scalar_type(), flood_type, [&] {
        BatchQKApplyRotary(
        static_cast<flood_type*>(q.data_ptr()), static_cast<flood_type*>(k.data_ptr()),
        static_cast<flood_type*>(q_rope.data_ptr()), static_cast<flood_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, interleave,
        rope_scale, rope_theta, torch_current_stream);
    });
}


/******************** apply_rope_inplace_kernel ********************/

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryInPlaceKernel(
    DType* __restrict__ q, DType* __restrict__ k, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, 
    float smooth_a, float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        if constexpr (interleave) {
          q_vec =
              vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        } else {
          q_vec = vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_ptr + tx * vec_size);
      }
    }
  } else {
    // apply rotary to k
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        if constexpr (interleave) {
          k_vec =
              vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        } else {
          k_vec = vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_ptr + tx * vec_size);
      }
    }
  }
}

/******************** apply_yarn_rope_inplace_kernel ********************/

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyYarnRotaryInPlaceKernel(
    DType* __restrict__ q, DType* __restrict__ k, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, float low,
    float high, float rope_rcp_scale, float rope_rcp_theta, float attention_factor) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    if constexpr (interleave) {
      freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
    } else {
      freq[i] = __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
    }

    // float smooth = freq[i] * smooth_a + smooth_b;
    // smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
    // freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];

    if (low == high){high = high + 0.001;}
    float extrapolation_factor;
    if constexpr (interleave) {
      extrapolation_factor = (float((tx * vec_size + i) / 2) - low) / (high - low);
    } else {
      extrapolation_factor = (float((tx * vec_size + i) % (rotary_dim / 2)) - low) / (high - low);
    }
    extrapolation_factor  = 1 - max(0.0f, min(1.0f, extrapolation_factor));
    freq[i] = (1 - extrapolation_factor) * (freq[i] * rope_rcp_scale) + extrapolation_factor * freq[i];
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        if constexpr (interleave) {
          q_vec =
              vec_apply_yarn_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, attention_factor, rotary_dim);//nor support
        } else {
          q_vec = vec_apply_yarn_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, attention_factor, rotary_dim);
        }
        q_vec.cast_store(q_ptr + tx * vec_size);
      }
    }
  } else {
    // apply rotary to k
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        if constexpr (interleave) {
          k_vec =
              vec_apply_yarn_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, attention_factor, rotary_dim);
        } else {
          k_vec = vec_apply_yarn_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, attention_factor, rotary_dim);
        }
        k_vec.cast_store(k_ptr + tx * vec_size);
      }
    }
  }
}


/******************** apply_rope_inplace_launcher ********************/

template <typename DType, typename IdType>
void BatchQKApplyRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim, 
                                      size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, 
                                      size_t k_stride_h, bool interleave, float rope_scale,
                                      float rope_theta, cudaStream_t stream = nullptr) {

    float rope_rcp_scale = 1.0f / rope_scale;
    float rope_rcp_theta = 1.0f / rope_theta;
    float smooth_a = 0.f;
    float smooth_b = 0.f;

    HEADDIM_SWITCH(head_dim, HEAD_DIM, [&] {


      constexpr uint32_t vec_size = std::max((uint32_t)(16 / sizeof(DType)), HEAD_DIM / 32);
      // constexpr uint32_t vec_size = 16 / sizeof(DType);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      // uint32_t num_threads = 128;
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);

      const bool INTERLEAVE = false; 

      BatchQKApplyRotaryInPlaceKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>
      <<<nblks, nthrs, 0, stream>>>(  q,
                                      k,
                                      indptr,
                                      offsets,
                                      batch_size,
                                      num_qo_heads,
                                      num_kv_heads,
                                      rotary_dim,
                                      q_stride_n,
                                      q_stride_h,
                                      k_stride_n,
                                      k_stride_h,
                                      smooth_a,
                                      smooth_b,
                                      rope_rcp_scale,
                                      rope_rcp_theta);
    });

}

/******************** apply_llama31_rope_inplace_launcher ********************/

template <typename DType, typename IdType>
void BatchQKApplyLlama31RotaryInPlace(
    DType* __restrict__ q, DType* __restrict__ k, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, 
    size_t k_stride_h, bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
    float high_freq_factor, float old_context_length, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  constexpr uint32_t HEAD_DIM = 128;

  // constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
  constexpr uint32_t vec_size = 16 / sizeof(DType);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  uint32_t num_threads = std::max(128U, bdx);
  uint32_t bdy = num_threads / bdx;
  dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
  dim3 nthrs(bdx, bdy);

  const bool INTERLEAVE = false; 
  
  BatchQKApplyRotaryInPlaceKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>
  <<<nblks, nthrs, 0, stream>>>( q,
                                 k,
                                 indptr,
                                 offsets,
                                 batch_size,
                                 num_qo_heads,
                                 num_kv_heads,
                                 rotary_dim,
                                 q_stride_n,
                                 q_stride_h,
                                 k_stride_n,
                                 k_stride_h,
                                 smooth_a,
                                 smooth_b,
                                 rope_rcp_scale,
                                 rope_rcp_theta);

}

/******************** apply_yarn_rope_inplace_launcher ********************/

template <typename DType, typename IdType>
void BatchQKApplyYarnRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n,
                                      size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
                                      bool interleave, float rope_scale, float rope_theta, float low, float high, float attention_factor,
                                      cudaStream_t stream = nullptr) {

    float rope_rcp_scale = 1.0f / rope_scale;
    float rope_rcp_theta = 1.0f / rope_theta;
    // float smooth_a = 0.f;
    // float smooth_b = 0.f;

    HEADDIM_SWITCH(head_dim, HEAD_DIM, [&] {
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        constexpr uint32_t vec_size = std::max((uint32_t)(16 / sizeof(DType)), HEAD_DIM / 32);
        // constexpr uint32_t vec_size = 16 / sizeof(DType);
        constexpr uint32_t bdx = HEAD_DIM / vec_size;
        uint32_t num_threads = std::max(128U, bdx);
        // uint32_t num_threads = 128;
        uint32_t bdy = num_threads / bdx;
        dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
        dim3 nthrs(bdx, bdy);
        BatchQKApplyYarnRotaryInPlaceKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>
        <<<nblks, nthrs, 0, stream>>>(  q,
                                        k,
                                        indptr,
                                        offsets,
                                        batch_size,
                                        num_qo_heads,
                                        num_kv_heads,
                                        rotary_dim,
                                        q_stride_n,
                                        q_stride_h,
                                        k_stride_n,
                                        k_stride_h,
                                        low,
                                        high,
                                        rope_rcp_scale,
                                        rope_rcp_theta,
                                        attention_factor);
      });
    });
}

void apply_rope_inplace(torch::Tensor& q, torch::Tensor& k, torch::Tensor& indptr,
                        torch::Tensor& offsets, int64_t rotary_dim, bool interleave, float rope_scale,
                        float rope_theta) {

  auto device = q.device();

  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);

  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(q.scalar_type(), flood_type, [&] {
    BatchQKApplyRotaryInPlace(
        static_cast<flood_type*>(q.data_ptr()), static_cast<flood_type*>(k.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, torch_current_stream);
  });
}

void apply_llama31_rope_inplace(torch::Tensor& q, torch::Tensor& k, torch::Tensor& indptr,
                                torch::Tensor& offsets, int64_t rotary_dim, bool interleave, float rope_scale,
                                float rope_theta, float low_freq_factor, float high_freq_factor,
                                float old_context_length) {

  auto device = q.device();
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);

  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(q.scalar_type(), flood_type, [&] {
    BatchQKApplyLlama31RotaryInPlace(
        static_cast<flood_type*>(q.data_ptr()), static_cast<flood_type*>(k.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor,
        old_context_length, torch_current_stream);
  });
}

void apply_yarn_rope_inplace(torch::Tensor& q, torch::Tensor& k, torch::Tensor& indptr,
                        torch::Tensor& offsets, int64_t rotary_dim, bool interleave, float rope_scale,
                        float rope_theta, float low, float high, float attention_factor) {

  auto device = q.device();

  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);

  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  indptr = indptr.to(torch::kInt32);
  offsets = offsets.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(q.scalar_type(), flood_type, [&] {
    BatchQKApplyYarnRotaryInPlace(
        static_cast<flood_type*>(q.data_ptr()), static_cast<flood_type*>(k.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h, k_stride_n,
        k_stride_h, interleave, rope_scale, rope_theta, low, high, attention_factor, torch_current_stream);
  });
}