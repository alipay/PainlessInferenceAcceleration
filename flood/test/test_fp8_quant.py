# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import flood_cuda


def torch_scaled_fp8_quant(x):
    x = x.to(torch.float32)
    scales = torch.abs(x).max(dim=-1, keepdim=True)[0]
    x = x / scales * 448.0
    x = x.to(torch.float8_e4m3fn)
    output_scales = scales / 448.0
    return x, output_scales


def test_dynamic_scaled_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int
) -> None:
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda:0")

    # reference
    ref_out, ref_scales = torch_scaled_fp8_quant(x)
    # kernel
    shape = x.shape
    ker_out = torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)

    ker_scales = torch.empty((shape[0], 1), device=x.device, dtype=torch.float32)
    flood_cuda.dynamic_per_token_scaled_fp8_quant(ker_out, x, ker_scales, None)

    torch.testing.assert_close(ker_scales, ref_scales, atol=0.01, rtol=0.01)
    torch.testing.assert_close(ker_out.float(), ref_out.float(), atol=0.01, rtol=0.01)


if __name__ == "__main__":
    test_dynamic_scaled_fp8_quant(256, 4096, torch.bfloat16, 42)
