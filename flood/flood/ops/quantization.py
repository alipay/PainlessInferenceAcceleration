
# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl

from flood.utils.benchmark import benchmark_func

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



if __name__ == '__main__':

    mode = 'dynamic'
    if mode == 'static':
        x = torch.randn((4096,8192),dtype=torch.bfloat16,device='cuda:0')
        static_scale = 0.05
        y = static_int8_quant(x, static_scale=static_scale,  block_size=1024)
        ref_y = torch.clamp((x/static_scale).to(torch.int32), -127, 127)
        error = (ref_y - y).abs().float().mean().item()
        print(f'error:{error:.3f}')

        benchmark_func(static_int8_quant,x,static_scale=static_scale,  block_size=1024)
        benchmark_func(deprecated_static_int8_quant,x,static_scale=static_scale,  block_size=1024)

    elif mode == 'dynamic':
        x = torch.randn((4096,8192),dtype=torch.bfloat16,device='cuda:0')
        y, scales = dynamic_int8_quant(x, block_size=1024)
        ref_scales = x.abs().amax(-1,keepdim=True)/127
        ref_y = torch.clamp((x/ref_scales).to(torch.int32), -127, 127)
        error = (ref_y - y).abs().float().mean().item()
        # print(ref_y)
        # print(y)
        print(f'error:{error:.3f}')

        benchmark_func(dynamic_int8_quant,x,block_size=1024)
        benchmark_func(deprecated_dynamic_int8_quant,x,block_size=1024)