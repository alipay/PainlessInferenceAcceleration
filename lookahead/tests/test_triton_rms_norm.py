# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import sys

# sys.path.append('../../lookahead')
import torch
import unittest
import triton

# from csrc.triton.rms_norm import rmsnorm_wrapper, rmsnorm_torch_precise
# import lookahead
# from lookahead.csrc.triton.rms_norm import rmsnorm_wrapper, rmsnorm_torch_precise
from lookahead.csrc import rmsnorm_wrapper, rmsnorm_torch_precise

class TestsCache(unittest.TestCase):

    def test_cell_get_one_branch(self):
        batch, seq_len, heads, dim = [1, 1000, 32, 128]

        embeddings = torch.randn([batch, seq_len, heads * dim], dtype=torch.float16, device="cuda")
        rms_weights = torch.randn([heads * dim], dtype=torch.float16, device="cuda")
        torch_output = rmsnorm_torch_precise(embeddings, rms_weights)
        triton_output = rmsnorm_wrapper(embeddings, rms_weights)

        diff = torch.abs(torch_output-triton_output).max()
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-3), diff)


if __name__ == '__main__':
    unittest.main()
