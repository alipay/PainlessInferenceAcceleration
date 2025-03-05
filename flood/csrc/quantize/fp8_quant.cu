// Copyright (c) Ant Financial Service Group and its affiliates.

#include "fp8_quant.cuh"
#include "cuda_type_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

// #include <cuda_fp8.h>
// using FP8_TYPE = __nv_fp8_e4m3;

template <typename scalar_t>
__global__ void scaled_fp8_quant_kernel(FP8_TYPE* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        float scale,
                                        int64_t num_elems
                                       ){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / scale;
  
  scaled_fp8_conversion_vec<scalar_t, true>(
      out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
}

template <typename scalar_t>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    FP8_TYPE* __restrict__ out, float* __restrict__ scale,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub,
    const int hidden_size) {
  float const min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);

  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const* __restrict__ token_input = &input[offset];
  FP8_TYPE* __restrict__ token_output = &out[offset];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float const x = flood_cast<float>(token_input[i]);
      absmax_val = max(absmax_val, fabs(x));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub) {
      token_scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = max(token_scale / FP8_E4M3_MAX, min_scaling_factor);
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  if (can_vectorize) {
    scaled_fp8_conversion_vec<scalar_t, false>(
        token_output, token_input, token_scale, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      token_output[i] = scaled_fp8_conversion<false>(
          flood_cast<float>(token_input[i]), token_scale);
    }
  }
}


void static_scaled_fp8_quant(torch::Tensor out,          // [..., d]
                             torch::Tensor const input,  // [..., d]
                             float scale) 
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // auto i = static_cast<float*>(input.data_ptr());
  // auto o = static_cast<FP8_TYPE*>(out.data_ptr());
  DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(
      input.scalar_type(), flood_type, [&] {
        scaled_fp8_quant_kernel<flood_type><<<grid, block, 0, stream>>>(
            static_cast<FP8_TYPE*>(out.data_ptr()), static_cast<flood_type*>(input.data_ptr()),
            scale, num_elems);
      });
}

// void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
//                               torch::Tensor const& input,  // [..., d]
//                               torch::Tensor& scale)        // [1]
// {
//   int64_t num_tokens = input.numel() / input.size(-1);
//   int64_t num_elems = input.numel();
//   dim3 grid(num_tokens);
//   dim3 block(1024);
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(
//       input.scalar_type(), flood_type, [&] {
//         segmented_max_reduction<flood_type><<<grid, block, 0, stream>>>(
//             scale.data_ptr<float>(), input.data_ptr<flood_type>(), num_elems);
//         scaled_fp8_quant_kernel<flood_type><<<grid, block, 0, stream>>>(
//             out.data_ptr<FP8_TYPE>(), input.data_ptr<flood_type>(),
//             scale.data_ptr<float>(), num_elems);
//       });
// }

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor out,          // [..., d]
    torch::Tensor const input,  // [..., d]
    torch::Tensor scales, std::optional<at::Tensor> const scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(
      input.scalar_type(), flood_type, [&] {
        dynamic_per_token_scaled_fp8_quant_kernel<flood_type>
            <<<grid, block, 0, stream>>>(
                static_cast<FP8_TYPE*>(out.data_ptr()), scales.data_ptr<float>(),
                static_cast<flood_type*>(input.data_ptr()),
                scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                hidden_size);
      });
}




__global__ void quant_to_fp8_and_update_cache_kernel(uint2* q_out, uint2* k_out,  uint2* v_out, uint4* query_states, uint4* key_states, uint4* value_states,  int* indices,  int group, int q_stride, int kv_stride) {
    
    int token_id = blockIdx.x;
    int tidx = threadIdx.x;
    int dim = blockDim.x;

    int slot_id =  indices[token_id];

    int kv_offset = token_id * kv_stride + tidx;

    __nv_fp8x4_e4m3 d[2];

    uint4 tmp1 = key_states[kv_offset];

    __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&tmp1);
    d[0] = __nv_fp8x4_e4m3(b[0], b[1]); 
    d[1] = __nv_fp8x4_e4m3(b[2], b[3]);

    uint2* e = reinterpret_cast<uint2*>(&d);

    k_out[slot_id * dim + tidx] = e[0];

    uint4 tmp2 = value_states[kv_offset];

    b = reinterpret_cast<__nv_bfloat162*>(&tmp2);
    d[0] = __nv_fp8x4_e4m3(b[0], b[1]); 
    d[1] = __nv_fp8x4_e4m3(b[2], b[3]);

    v_out[slot_id * dim + tidx] = e[0];

    int q_offset = token_id * q_stride + tidx*group;
    // #progma unroll
    for(int i=0;i<group;++i){
        uint4 tmp3 = query_states[q_offset+i];
        b = reinterpret_cast<__nv_bfloat162*>(&tmp3);
        d[0] = __nv_fp8x4_e4m3(b[0], b[1]); 
        d[1] = __nv_fp8x4_e4m3(b[2], b[3]);
        q_out[token_id * dim * group + tidx*group+i] = e[0];
    }

}

void quant_to_fp8_and_update_cache(torch::Tensor& q_out, 
                  torch::Tensor& k_out, 
                  torch::Tensor& v_out,
                  torch::Tensor& query_states,  
                  torch::Tensor& key_states, 
                  torch::Tensor& value_states,  
                  torch::Tensor& indices, 
                  int tok, int group, int kv_dim, int q_stride, int kv_stride) 
{

    dim3 blocks(tok);
    dim3 threads(kv_dim/8);



    // const at::cuda::OptionalCUDAGuard device_guard(device_of(k_out));
    const cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();
    
    quant_to_fp8_and_update_cache_kernel<<<blocks, threads, 0, current_stream>>>(
        static_cast<uint2*>(q_out.data_ptr()), 
        static_cast<uint2*>(k_out.data_ptr()), 
        static_cast<uint2*>(v_out.data_ptr()), 
        static_cast<uint4*>(query_states.data_ptr()), 
        static_cast<uint4*>(key_states.data_ptr()), 
        static_cast<uint4*>(value_states.data_ptr()), 
        static_cast<int32_t*>(indices.data_ptr()), 
        group, q_stride, kv_stride);
}
