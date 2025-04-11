
# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional, Tuple, Union
import torch
import triton
import triton.language as tl

import flood_cuda

@triton.jit
def deprecated_static_int8_quant_kernel(x_ptr, y_ptr, static_scale, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    sms = tl.num_programs(0)
    n_token = (M-1) // sms + 1
    n_block = (N-1) // BLOCK_SIZE + 1 
    indices = tl.arange(0, BLOCK_SIZE)
    for i in range(n_token):
        if pid*n_token + i < M:
            for j in range(n_block):
                offs = (pid*n_token+i)*N + j*BLOCK_SIZE + indices
                x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
                y = x.to(tl.float32) / static_scale
                y = y.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def deprecated_static_int8_quant(x: torch.Tensor, static_scale: float,  block_size: int = 1024) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (sms, )  # noqa: E731
    deprecated_static_int8_quant_kernel[grid](x, y, static_scale, M, N, 
                            BLOCK_SIZE=block_size, 
                            num_stages=5,
                            num_warps=16
                            )
    return y


@triton.jit
def static_int8_quant_kernel(x_ptr, y_ptr, static_scale, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = (N-1) // BLOCK_SIZE + 1 
    indices = tl.arange(0, BLOCK_SIZE)
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) / static_scale
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def static_int8_quant(x: torch.Tensor, static_scale: float,  block_size: int = 1024) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    grid = lambda meta: (M, )  
    static_int8_quant_kernel[grid](x, y, static_scale, M, N, 
                            BLOCK_SIZE=block_size, 
                            num_stages=5,
                            num_warps=16
                            )
    return y


@triton.jit
def deprecated_dynamic_int8_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    sms = tl.num_programs(0)
    n_token = (M-1) // sms + 1
    n_block = (N-1) // BLOCK_SIZE + 1 
    indices = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(n_token, num_stages=5):
        if pid*n_token + i < M:
            max_val = 0.0
            for j in range(n_block):
                offs = (pid*n_token+i)*N + j*BLOCK_SIZE + indices
                x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
                max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
            scale = max_val/127
            tl.store(s_ptr + pid*n_token+i, scale)
            for j in tl.range(n_block, num_stages=1):
                offs = (pid*n_token+i)*N + j*BLOCK_SIZE + indices
                x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
                y = x.to(tl.float32) / scale
                y = y.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def deprecated_dynamic_int8_quant(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M,), dtype=torch.float32,device=x.device)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (sms, )  # noqa: E731
    deprecated_dynamic_int8_quant_kernel[grid](x, y, scales, M, N, 
                            BLOCK_SIZE=block_size, 
                            num_stages=5,
                            num_warps=16
                            )
    return y, scales




@triton.jit
def dynamic_int8_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = (N-1) // BLOCK_SIZE + 1 
    indices = tl.arange(0, BLOCK_SIZE)
    max_val = 0.0
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
    scale = max_val/127
    tl.store(s_ptr + pid, scale)
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) / scale
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def dynamic_int8_quant(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M,), dtype=torch.float32,device=x.device)
    grid = lambda meta: (M, )  # noqa: E731
    dynamic_int8_quant_kernel[grid](x, y, scales, M, N, 
                            BLOCK_SIZE=block_size, 
                            num_stages=5,
                            num_warps=16
                            )
    return y, scales



# Some triton kernels for tilewise and blockwise quantization are from the link below with modification:
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def block_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)), 1e-10) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def block_quant(x: torch.Tensor, dtype=torch.float8_e4m3fn, block_size: int = 128) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(x.size(-2) // block_size, x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))  # noqa: E731
    block_quant_kernel[grid](x, y, s, M, N, 
                             BLOCK_SIZE=block_size, 
                             num_stages=6,
                             num_warps=8)
    return y, s



@triton.jit
def tile_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * N + tl.arange(0, K*BLOCK_SIZE)
    n = tl.cdiv(N, K*BLOCK_SIZE)
    soffs = pid * n + tl.arange(0, K)
    for i in range(n):
        x = tl.load(x_ptr + offs, mask=offs < N).to(tl.float32)
        x = tl.reshape(x, (K, BLOCK_SIZE))
        s = tl.maximum(tl.max(tl.abs(x),1), 1e-10) / 448.0
        s = tl.floor(tl.log2(s) + 0.5)
        s = tl.exp2(s)
        y = x / s[:,None]
        y = y.to(y_ptr.dtype.element_ty)
        y = tl.reshape(y, (K*BLOCK_SIZE,))
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + soffs, s, mask=soffs<n)
        offs += K*BLOCK_SIZE
        soffs += K




def tile_quant(
    x: torch.Tensor, dtype=torch.float8_e4m3fn, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    M, N= x.shape
    y = torch.empty_like(x, dtype=dtype)
    s = torch.empty(M, N // block_size, device=x.device, dtype=torch.float32)
    K = 16
    grid = lambda meta: (M,)  # noqa: E731
    tile_quant_kernel[grid](x, y, s, M, N, block_size, K, num_stages=5, num_warps=4)
    return y, s



def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    assert (input.ndim == 2)
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    out_dtype: torch.dtype = torch.float8_e4m3fn
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        scale = torch.empty((shape[0], 1),
                            device=input.device,
                            dtype=torch.float32)
        flood_cuda.dynamic_per_token_scaled_fp8_quant(output, input, scale,
                                                        scale_ub)
    else:
        assert scale.numel() == 1
        flood_cuda.static_scaled_fp8_quant(output, input, scale)

    return output, scale



