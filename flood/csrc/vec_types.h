//Adapt (heavily) from https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/vec_dtypes.cuh
// Modified by Chen Liang
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
// Copyright (c) Ant Financial Service Group and its affiliates.



#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <type_traits>

#include "cuda_type_utils.h"

#define FLOOD_INLINE inline __attribute__((always_inline)) __device__

/**********  vec cast ***********/
template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  FLOOD_INLINE static void cast(dst_t* dst, const src_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
    //   dst[i] = (dst_t)src[i];
      dst[i] = flood_cast<dst_t>(src[i]);
    }
  }
};

template <>
struct vec_cast<float, half> {
  template <size_t vec_size>
  FLOOD_INLINE static void cast(float* dst, const half* src) {
    if constexpr (vec_size == 1) {
      dst[0] = flood_cast<float>(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
      }
    }
  }
};

template <>
struct vec_cast<float, nv_bfloat16> {
  template <size_t vec_size>
  FLOOD_INLINE static void cast(float* dst, const nv_bfloat16* src) {
    if constexpr (vec_size == 1) {
      dst[0] = flood_cast<float>(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __bfloat1622float2(((nv_bfloat162*)src)[i]);
      }
    }
  }
};

/********** vec_t **********/
template <typename float_t, size_t vec_size>
struct vec_t {
  FLOOD_INLINE float_t& operator[](size_t i);
  FLOOD_INLINE const float_t& operator[](size_t i) const;
  FLOOD_INLINE void fill(float_t val);
  FLOOD_INLINE void load(const float_t* ptr);
  FLOOD_INLINE void store(float_t* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr);
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const;
  FLOOD_INLINE static void memcpy(float_t* dst, const float_t* src);
  FLOOD_INLINE float_t* ptr();
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLOOD_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const vec_t<src_float_t, vec_size>& src) {
  vec_cast<tgt_float_t, src_float_t>::cast<vec_size>(
      dst.ptr(), const_cast<vec_t<src_float_t, vec_size>*>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLOOD_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const src_float_t* src_ptr) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLOOD_INLINE void cast_store_impl(tgt_float_t* dst_ptr,
                                       const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

/********** vec_t<float> **********/
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  FLOOD_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLOOD_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLOOD_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLOOD_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  FLOOD_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  FLOOD_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)dst)[i] = ((float4*)src)[i];
    }
  }
};


/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  FLOOD_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLOOD_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLOOD_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLOOD_INLINE void fill(half val);
  FLOOD_INLINE void load(const half* ptr);
  FLOOD_INLINE void store(half* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLOOD_INLINE static void memcpy(half* dst, const half* src);
};

FLOOD_INLINE void vec_t<half, 1>::fill(half val) { data = val; }

FLOOD_INLINE void vec_t<half, 1>::load(const half* ptr) { data = *ptr; }

FLOOD_INLINE void vec_t<half, 1>::store(half* ptr) const { *ptr = data; }

FLOOD_INLINE void vec_t<half, 1>::memcpy(half* dst, const half* src) { *dst = *src; }

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  FLOOD_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLOOD_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLOOD_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLOOD_INLINE void fill(half val);
  FLOOD_INLINE void load(const half* ptr);
  FLOOD_INLINE void store(half* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLOOD_INLINE static void memcpy(half* dst, const half* src);
};

FLOOD_INLINE void vec_t<half, 2>::fill(half val) { data = make_half2(val, val); }

FLOOD_INLINE void vec_t<half, 2>::load(const half* ptr) { data = *((half2*)ptr); }

FLOOD_INLINE void vec_t<half, 2>::store(half* ptr) const { *((half2*)ptr) = data; }

FLOOD_INLINE void vec_t<half, 2>::memcpy(half* dst, const half* src) {
  *((half2*)dst) = *((half2*)src);
}

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  FLOOD_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLOOD_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLOOD_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLOOD_INLINE void fill(half val);
  FLOOD_INLINE void load(const half* ptr);
  FLOOD_INLINE void store(half* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(half* dst, const half* src);
};

FLOOD_INLINE void vec_t<half, 4>::fill(half val) {
  *(half2*)(&data.x) = make_half2(val, val);
  *(half2*)(&data.y) = make_half2(val, val);
}

FLOOD_INLINE void vec_t<half, 4>::load(const half* ptr) { data = *((uint2*)ptr); }

FLOOD_INLINE void vec_t<half, 4>::store(half* ptr) const { *((uint2*)ptr) = data; }

FLOOD_INLINE void vec_t<half, 4>::memcpy(half* dst, const half* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// half x 8 or more

template <size_t vec_size>
struct vec_t<half, vec_size> {
  uint4 data[vec_size / 8];
  FLOOD_INLINE half& operator[](size_t i) { return ((half*)data)[i]; }
  FLOOD_INLINE const half& operator[](size_t i) const { return ((const half*)data)[i]; }
  FLOOD_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLOOD_INLINE void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(half2*)(&(data[i].x)) = make_half2(val, val);
      *(half2*)(&(data[i].y)) = make_half2(val, val);
      *(half2*)(&(data[i].z)) = make_half2(val, val);
      *(half2*)(&(data[i].w)) = make_half2(val, val);
    }
  }
  FLOOD_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLOOD_INLINE void store(half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(half* dst, const half* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;
  FLOOD_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  FLOOD_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  FLOOD_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLOOD_INLINE void fill(nv_bfloat16 val);
  FLOOD_INLINE void load(const nv_bfloat16* ptr);
  FLOOD_INLINE void store(nv_bfloat16* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

FLOOD_INLINE void vec_t<nv_bfloat16, 1>::fill(nv_bfloat16 val) { data = val; }

FLOOD_INLINE void vec_t<nv_bfloat16, 1>::load(const nv_bfloat16* ptr) { data = *ptr; }

FLOOD_INLINE void vec_t<nv_bfloat16, 1>::store(nv_bfloat16* ptr) const { *ptr = data; }

FLOOD_INLINE void vec_t<nv_bfloat16, 1>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *dst = *src;
}

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;

  FLOOD_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  FLOOD_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  FLOOD_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLOOD_INLINE void fill(nv_bfloat16 val);
  FLOOD_INLINE void load(const nv_bfloat16* ptr);
  FLOOD_INLINE void store(nv_bfloat16* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

FLOOD_INLINE void vec_t<nv_bfloat16, 2>::fill(nv_bfloat16 val) {
  data = make_bfloat162(val, val);
}

FLOOD_INLINE void vec_t<nv_bfloat16, 2>::load(const nv_bfloat16* ptr) {
  data = *((nv_bfloat162*)ptr);
}

FLOOD_INLINE void vec_t<nv_bfloat16, 2>::store(nv_bfloat16* ptr) const {
  *((nv_bfloat162*)ptr) = data;
}

FLOOD_INLINE void vec_t<nv_bfloat16, 2>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *((nv_bfloat162*)dst) = *((nv_bfloat162*)src);
}

// nv_bfloat16 x 4

template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;

  FLOOD_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)(&data))[i]; }
  FLOOD_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  FLOOD_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLOOD_INLINE void fill(nv_bfloat16 val);
  FLOOD_INLINE void load(const nv_bfloat16* ptr);
  FLOOD_INLINE void store(nv_bfloat16* ptr) const;
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

FLOOD_INLINE void vec_t<nv_bfloat16, 4>::fill(nv_bfloat16 val) {
  *(nv_bfloat162*)(&data.x) = make_bfloat162(val, val);
  *(nv_bfloat162*)(&data.y) = make_bfloat162(val, val);
}

FLOOD_INLINE void vec_t<nv_bfloat16, 4>::load(const nv_bfloat16* ptr) {
  data = *((uint2*)ptr);
}

FLOOD_INLINE void vec_t<nv_bfloat16, 4>::store(nv_bfloat16* ptr) const {
  *((uint2*)ptr) = data;
}

FLOOD_INLINE void vec_t<nv_bfloat16, 4>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// nv_bfloat16 x 8 or more

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  uint4 data[vec_size / 8];

  FLOOD_INLINE nv_bfloat16& operator[](size_t i) { return ((nv_bfloat16*)data)[i]; }
  FLOOD_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)data)[i];
  }
  FLOOD_INLINE nv_bfloat16* ptr() { return reinterpret_cast<nv_bfloat16*>(&data); }
  FLOOD_INLINE void fill(nv_bfloat16 val) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(nv_bfloat162*)(&(data[i].x)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].y)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].z)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].w)) = make_bfloat162(val, val);
    }
  }
  FLOOD_INLINE void load(const nv_bfloat16* ptr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLOOD_INLINE void store(nv_bfloat16* ptr) const {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLOOD_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLOOD_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLOOD_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLOOD_INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};
