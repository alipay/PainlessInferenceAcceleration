// Copyright (c) Ant Financial Service Group and its affiliates.


#include <torch/extension.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_type_utils.h"


template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
//   return (T)(((float)x) / (1.0f + expf((float)-x)));
  float val   = flood_cast<float>(x);
  val         = val / (1.0f + __expf(-val));
  return flood_cast<T>(val);
//   return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {

        const int64_t token_idx = blockIdx.x;
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = input[token_idx * 2 * d + idx];
            const scalar_t y = input[token_idx * 2 * d + d + idx];
            // out[token_idx * d + idx] = ACT_FN(x) * y;
            // out[token_idx * d + idx] = x * y;
            out[token_idx * d + idx] = silu_kernel(x) * y;
            // out[token_idx * d + idx] = __hmul(ACT_FN(x), y);
  }
}

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
    int d = input.size(-1) / 2; 
    int64_t num_tokens = input.numel() / input.size(-1);
    dim3 grid(num_tokens);                                                 
    dim3 block(std::min(d, 1024));   
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(input.scalar_type(), flood_type, [&] {
        act_and_mul_kernel<flood_type>
        <<<grid, block, 0, stream>>>(static_cast<flood_type*>(out.data_ptr()), 
                                     static_cast<flood_type*>(input.data_ptr()), d);
    });                                      
}