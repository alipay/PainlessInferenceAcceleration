// Copyright (c) Ant Financial Service Group and its affiliates.

#include <cuda_runtime.h> 
#include <torch/all.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cuda_fp8.h>


#include "cuda_type_utils.h"

__global__ void update_cache_kernel(uint4* k_out,  uint4* v_out,  uint4* key_states, uint4* value_states,  int* indices,  int stride) {
    
    int token_id = blockIdx.x;
    int tid = threadIdx.x;
    int dim = blockDim.x;

    // int slot_id =  __ldg(indices+token_id);
    int slot_id =  indices[token_id];

    int offset = token_id * stride +  tid;

    // uint4 tmp1 = __ldg(key_states + offset);
    // uint4 tmp2 = __ldg(value_states + offset);

    // uint4 tmp1 = key_states[offset];
    // uint4 tmp2 = value_states[offset];

    k_out[slot_id * dim + tid] = key_states[offset];
    v_out[slot_id * dim + tid] = value_states[offset];

}

void update_cache(torch::Tensor& k_out, 
                  torch::Tensor& v_out, 
                  torch::Tensor& key_states, 
                  torch::Tensor& value_states,  
                  torch::Tensor& indices, 
                  int tok, int dim, int stride) 
{

    dim3 blocks(tok);
    dim3 threads(dim/8);

    // const at::cuda::OptionalCUDAGuard device_guard(device_of(k_out));
    const cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();
    
    update_cache_kernel
    <<<blocks, threads, 0, current_stream>>>(static_cast<uint4*>(k_out.data_ptr()), 
                                             static_cast<uint4*>(v_out.data_ptr()), 
                                             static_cast<uint4*>(key_states.data_ptr()), 
                                             static_cast<uint4*>(value_states.data_ptr()), 
                                             static_cast<int32_t*>(indices.data_ptr()), stride);
}



__global__ void update_fusion_cache_kernel(uint4* kv_out,  uint4* kv_states, int* indices, int stride) {
    
    int token_id = blockIdx.x;
    int tid = threadIdx.x;
    int dim = blockDim.x;

    // int slot_id =  __ldg(indices+token_id);
    int slot_id =  indices[token_id];

    int offset = token_id * stride +  tid;

    // uint4 tmp1 = __ldg(key_states + offset);
    // uint4 tmp2 = __ldg(value_states + offset);

    // uint4 tmp1 = key_states[offset];
    // uint4 tmp2 = value_states[offset];

    kv_out[slot_id * dim + tid] = kv_states[offset];

}

void update_fusion_cache(torch::Tensor& kv_out, 
                  torch::Tensor& kv_states, 
                  torch::Tensor& indices, 
                  int tok, int dim, int stride) 
{

    dim3 blocks(tok);
    dim3 threads(dim/8);

    // const at::cuda::OptionalCUDAGuard device_guard(device_of(k_out));
    const cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();
    
    update_fusion_cache_kernel
    <<<blocks, threads, 0, current_stream>>>(static_cast<uint4*>(kv_out.data_ptr()), 
                                             static_cast<uint4*>(kv_states.data_ptr()), 
                                             static_cast<int32_t*>(indices.data_ptr()), stride);
}


