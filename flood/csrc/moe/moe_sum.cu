// The file is copied from https://github.com/vllm-project/vllm/blob/main/csrc/moe/moe_align_sum_kernels.cu


#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/util_type.cuh>
#include <cub/cub.cuh>

#include "cuda_type_utils.h"

template <typename scalar_t>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d, const int topk) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = flood_cast<scalar_t>(0.0f);
#pragma unroll
    for (int k = 0; k < topk; ++k) {
      x += input[token_idx * topk * d + k * d + idx];
    }
    out[token_idx * d + idx] = x;
  }
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& out)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const int num_tokens = out.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (topk < 5) {
    DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(input.scalar_type(), flood_type, [&] {
      moe_sum_kernel<flood_type>
      <<<grid, block, 0, stream>>>(static_cast<flood_type*>(out.data_ptr()), 
                                    static_cast<flood_type*>(input.data_ptr()), hidden_size, topk);
    });   
  } else {
    at::sum_out(out, input, 1);
  }      

}