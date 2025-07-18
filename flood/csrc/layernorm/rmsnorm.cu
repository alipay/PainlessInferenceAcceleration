/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/*
Adapted (Heavily) from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/layernorm_kernels.cu

Modified by Chen Liang
*/

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_type_utils.h"
// #include "reduction.cuh"
#include "rmsnorm.h"

#include <cub/util_type.cuh>
#include <cub/cub.cuh>

// static inline __device__ float to_float(float src)
// {
//     return src;
// }

template<typename T>
__global__ void RMSNormkernel(
    const T* __restrict input, const T* __restrict gamma, T* output, const float layernorm_eps, int m, int n, int input_stride, int output_stride)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        // float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        // float diff = (float)(ldg(&input[blockIdx.x * n + i]));
        float diff = flood_cast<float>(input[blockIdx.x * input_stride + blockIdx.y * n + i]);
        local_var_sum += diff * diff;
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(local_var_sum, cub::Sum{}, blockDim.x);
    // variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float x = flood_cast<float>(input[blockIdx.x * input_stride + blockIdx.y * n + i]);
        output[blockIdx.x * output_stride + blockIdx.y * n + i] =flood_cast<T>(x * s_variance) * gamma[i];
        // output[blockIdx.x * n + i] =
        //     flood_cast<T>((flood_cast<float>(input[blockIdx.x * n + i]) * s_variance) * flood_cast<float>(gamma[i]));
    }
}


template<typename T>
void RMSNorm(T*           out,
             const T*     input,
             const T*     gamma,
             // const T*     beta,
             const float  layernorm_eps,
             const int    m,
             const int    n,
             const int    k,
             const int    input_stride,
             const int    output_stride)
{
    dim3 grid(m, k);
    dim3 block(min(n, 1024));

    // block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

    /* should pay attention to the rsqrt precision*/
    RMSNormkernel<T><<<grid, block, 0, stream>>>(input, gamma, out, layernorm_eps, m, n, input_stride, output_stride);  // For gpt-3
}

template void RMSNorm(half*           out,
                      const half*     input,
                      const half*     gamma,
                      // const half*     beta,
                      const float  layernorm_eps,
                      const int    m,
                      const int    n,
                      const int    k,
                      const int    input_stride,
                      const int    output_stride);

template void RMSNorm(__nv_bfloat16*           out,
                      const __nv_bfloat16*     input,
                      const __nv_bfloat16*     gamma,
                      // const half*     beta,
                      const float  layernorm_eps,
                      const int    m,
                      const int    n,
                      const int    k,
                      const int    input_stride,
                      const int    output_stride);

// template void RMSNorm(float*           out,
//                               const float*     input,
//                               const float*     gamma,
//                               // const half*     beta,
//                               const float  layernorm_eps,
//                               const int    m,
//                               const int    n);


// input b, n, c
void rmsnorm(
    torch::Tensor& _input,
    torch::Tensor& _gamma,
    torch::Tensor& _out,
    float eps)
{
    // int m = _input.size(0) * _input.size(1);
    // int n = _input.size(2);
    int n = _input.size(-1);
    int m = _input.size(0);
    int k = _input.numel() / (m * n);
    int input_stride = _input.stride(0);
    int output_stride = _out.stride(0);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_input));

    // auto input = reinterpret_cast<half*>(_input.data_ptr<at::Half>());
    // auto gamma = reinterpret_cast<half*>(_gamma.data_ptr<at::Half>());
    // auto out = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

    DISPATCH_PYTORCH_DTYPE_TO_FLOOD_TYPE(_input.scalar_type(), flood_type, [&] {
            RMSNorm(static_cast<flood_type*>(_out.data_ptr()), 
                    static_cast<flood_type*>(_input.data_ptr()), 
                    static_cast<flood_type*>(_gamma.data_ptr()), 
                    eps, 
                    m, 
                    n,
                    k,
                    input_stride,
                    output_stride);
        // return true;
    });
}
