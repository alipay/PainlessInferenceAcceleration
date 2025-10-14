# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import os
import json
import functools
from typing import Optional, Tuple, Union, Any
import torch
import triton
import triton.language as tl

try:
    import flood_cuda
except ImportError:
    pass

@triton.jit
def deprecated_static_int8_quant_kernel(
    x_ptr, y_ptr, static_scale, M, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    sms = tl.num_programs(0)
    n_token = (M - 1) // sms + 1
    n_block = (N - 1) // BLOCK_SIZE + 1
    indices = tl.arange(0, BLOCK_SIZE)
    for i in range(n_token):
        if pid * n_token + i < M:
            for j in range(n_block):
                offs = (pid * n_token + i) * N + j * BLOCK_SIZE + indices
                x = tl.load(x_ptr + offs, mask=j * BLOCK_SIZE + indices < N, other=0)
                y = x.to(tl.float32) / static_scale
                y = y.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y, mask=j * BLOCK_SIZE + indices < N)


def deprecated_static_int8_quant(
    x: torch.Tensor, static_scale: float, block_size: int = 1024
) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (sms,)  # noqa: E731
    deprecated_static_int8_quant_kernel[grid](
        x, y, static_scale, M, N, BLOCK_SIZE=block_size, num_stages=5, num_warps=16
    )
    return y


@triton.jit
def static_int8_quant_kernel(
    x_ptr, y_ptr, static_scale, M, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_block = (N - 1) // BLOCK_SIZE + 1
    indices = tl.arange(0, BLOCK_SIZE)
    for j in range(n_block):
        offs = pid * N + j * BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j * BLOCK_SIZE + indices < N, other=0)
        y = x.to(tl.float32) / static_scale
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y, mask=j * BLOCK_SIZE + indices < N)


def static_int8_quant(
    x: torch.Tensor, static_scale: float, block_size: int = 1024
) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    grid = lambda meta: (M,)
    static_int8_quant_kernel[grid](
        x, y, static_scale, M, N, BLOCK_SIZE=block_size, num_stages=5, num_warps=16
    )
    return y


@triton.jit
def deprecated_dynamic_int8_quant_kernel(
    x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    sms = tl.num_programs(0)
    n_token = (M - 1) // sms + 1
    n_block = (N - 1) // BLOCK_SIZE + 1
    indices = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(n_token, num_stages=5):
        if pid * n_token + i < M:
            max_val = 0.0
            for j in range(n_block):
                offs = (pid * n_token + i) * N + j * BLOCK_SIZE + indices
                x = tl.load(x_ptr + offs, mask=j * BLOCK_SIZE + indices < N, other=0)
                max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
            scale = max_val / 127
            tl.store(s_ptr + pid * n_token + i, scale)
            for j in tl.range(n_block, num_stages=1):
                offs = (pid * n_token + i) * N + j * BLOCK_SIZE + indices
                x = tl.load(x_ptr + offs, mask=j * BLOCK_SIZE + indices < N, other=0)
                y = x.to(tl.float32) / scale
                y = y.to(y_ptr.dtype.element_ty)
                tl.store(y_ptr + offs, y, mask=j * BLOCK_SIZE + indices < N)


def deprecated_dynamic_int8_quant(
    x: torch.Tensor, block_size: int = 1024
) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M,), dtype=torch.float32, device=x.device)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (sms,)  # noqa: E731
    deprecated_dynamic_int8_quant_kernel[grid](
        x, y, scales, M, N, BLOCK_SIZE=block_size, num_stages=5, num_warps=16
    )
    return y, scales


@triton.jit
def dynamic_int8_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = (N - 1) // BLOCK_SIZE + 1
    indices = tl.arange(0, BLOCK_SIZE)
    max_val = 0.0
    for j in range(n_block):
        offs = pid * N + j * BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j * BLOCK_SIZE + indices < N, other=0)
        max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
    scale = max_val / 127
    tl.store(s_ptr + pid, scale)
    for j in range(n_block):
        offs = pid * N + j * BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j * BLOCK_SIZE + indices < N, other=0)
        y = x.to(tl.float32) / scale
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y, mask=j * BLOCK_SIZE + indices < N)


def dynamic_int8_quant(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M,), dtype=torch.float32, device=x.device)
    grid = lambda meta: (M,)  # noqa: E731
    dynamic_int8_quant_kernel[grid](
        x, y, scales, M, N, BLOCK_SIZE=block_size, num_stages=5, num_warps=16
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


def block_quant(
    x: torch.Tensor, dtype=torch.float8_e4m3fn, block_size: int = 128
) -> torch.Tensor:
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(
        x.size(-2) // block_size, x.size(-1) // block_size, dtype=torch.float32
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )  # noqa: E731
    block_quant_kernel[grid](
        x, y, s, M, N, BLOCK_SIZE=block_size, num_stages=6, num_warps=8
    )
    return y, s


@triton.jit
def tile_quant_kernel(
    x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr, K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * N + tl.arange(0, K * BLOCK_SIZE)
    n = tl.cdiv(N, K * BLOCK_SIZE)
    soffs = pid * n + tl.arange(0, K)
    for i in range(n):
        x = tl.load(x_ptr + offs, mask=offs < N).to(tl.float32)
        x = tl.reshape(x, (K, BLOCK_SIZE))
        s = tl.maximum(tl.max(tl.abs(x), 1), 1e-10) / 448.0
        s = tl.floor(tl.log2(s) + 0.5)
        s = tl.exp2(s)
        y = x / s[:, None]
        y = y.to(y_ptr.dtype.element_ty)
        y = tl.reshape(y, (K * BLOCK_SIZE,))
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + soffs, s, mask=soffs < n)
        offs += K * BLOCK_SIZE
        soffs += K


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


# Adapted from https://github.com/vllm-project/vllm/blob/a2480251ec92ba2a849464dde48db8a2b7f6ef81/vllm/model_executor/layers/quantization/utils/fp8_utils.py
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    out_q: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty_like(x, device=x.device, dtype=dtype)

    M = x.numel() // group_size
    N = group_size
    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[1],
        x.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=False,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def tile_quant(
    x: torch.Tensor, dtype=torch.float8_e4m3fn, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    M, N = x.shape
    y = torch.empty_like(x, dtype=dtype)
    s = torch.empty(M, N // block_size, device=x.device, dtype=torch.float32)
    K = 16
    grid = lambda meta: (M,)  # noqa: E731
    tile_quant_kernel[grid](x, y, s, M, N, block_size, K, num_stages=5, num_warps=4)
    return y, s


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
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
    assert input.ndim == 2
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    out_dtype: torch.dtype = torch.float8_e4m3fn
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
        flood_cuda.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
    else:
        assert scale.numel() == 1
        flood_cuda.static_scaled_fp8_quant(output, input, scale)

    return output, scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: Union[None, torch.dtype, str],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if quant_dtype == torch.float8_e4m3fn:
        if block_shape is None:
            return scaled_fp8_quant(A, A_scale)
        else:
            assert len(block_shape) == 2
            _, block_k = block_shape[0], block_shape[1]
            return per_token_group_quant_fp8(A, group_size=block_k)
    else:
        return A, A_scale
