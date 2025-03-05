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
Adapt from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/reduce_kernel_utils.cuh
Modified by Chen Liang
*/

#pragma once
#include <assert.h>
#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
#include <cooperative_groups/reduce.h>
#else
#include <cooperative_groups.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <type_traits>

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    return __float2bfloat16( __bfloat162float(x) + __bfloat162float(y) );
}

template<typename T>
inline __device__ T add(T a, T b) {
    return a + b;
}

// template<>
// inline __device__ half2 add(half2 a, half2 b) {
//     return __hadd2(a, b);
// }

template<>
inline __device__ half add(half a, half b) {
    return __hadd(a, b);
}

// template<>
// inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
//     return bf16hadd2(a, b);
// }

template<>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return bf16hadd(a, b);
}

// inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, float b) {
//     return bf16hadd(a, __float2bfloat16(b));
// }

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));  //__shfl_sync bf16 return float when sm < 80
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;
    int                 wid  = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}


template<typename T>
__device__ __forceinline__ T clamp_inf_for_half(const float input)
{
    return input;
}

template<>
__device__ __forceinline__ half clamp_inf_for_half(const float input)
{
    return input > 0.0f ? __float2half(min(input, HALF_FLT_MAX - 1000)) : __float2half(max(input, -HALF_FLT_MAX + 1000));
}

template<>
__device__ __forceinline__ __nv_bfloat16 clamp_inf_for_half(const float input)
{
    return __float2bfloat16(input);
    // return input > 0.0f ? __float2bfloat16(min(input, HALF_FLT_MAX - 1000)) : __float2half(max(input, -HALF_FLT_MAX + 1000));
}