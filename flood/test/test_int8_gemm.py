# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from flood.ops.gemm import static_int8_gemm_nt, dynamic_int8_gemm_nt


mode = "dynamic"
if mode == "static":
    M, N, K = 456, 4096, 8192
    x = (10 * torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")).to(torch.int8)
    w = (10 * torch.randn((N, K), dtype=torch.bfloat16, device="cuda:0")).to(torch.int8)
    y = static_int8_gemm_nt(x, w, 1.0, 1.0, torch.bfloat16)
    ref_y = x.float() @ w.float().t()
    error = (y.float() - ref_y).abs().mean().item()
    rate = error / ref_y.abs().mean().item()
    print(f"error:{error:.3f} rate:{rate:.3f}")
else:
    M, N, K = 457, 4096, 8192
    x = (10 * torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")).to(torch.int8)
    w = (10 * torch.randn((N, K), dtype=torch.bfloat16, device="cuda:0")).to(torch.int8)
    xs = torch.rand((M,), dtype=torch.float32, device="cuda:0")
    ws = torch.rand((N,), dtype=torch.float32, device="cuda:0")
    y = dynamic_int8_gemm_nt(x, w, xs, ws, torch.bfloat16)
    ref_y = (x.float() * xs[:, None]) @ (w.float().t() * ws)
    error = (y.float() - ref_y).abs().mean().item()
    rate = error / ref_y.abs().mean().item()
    print(f"error:{error:.3f} rate:{rate:.3f}")
