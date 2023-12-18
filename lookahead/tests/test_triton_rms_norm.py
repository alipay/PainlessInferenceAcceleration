# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import sys

sys.path.append('../../lookahead')
import torch
import unittest
import triton

from csrc.triton.rms_norm import rmsnorm_wrapper, rmsnorm_torch


class TestsCache(unittest.TestCase):

    def test_cell_get_one_branch(self):
        batch, seq_len, heads, dim = [1, 1000, 32, 128]

        embeddings = torch.randn([batch, seq_len, heads * dim], dtype=torch.float16, device="cuda")
        rms_weights = torch.randn([heads * dim], dtype=torch.float16, device="cuda")

        self.assertTrue(
            torch.allclose(rmsnorm_torch(embeddings, rms_weights), rmsnorm_wrapper(embeddings, rms_weights)), 'close')

        print("triton rmsnorm", triton.testing.do_bench(lambda: rmsnorm_wrapper(x=embeddings, rms_weights=rms_weights)))
        print("rmsnorm pytorch", triton.testing.do_bench(lambda: rmsnorm_torch(x=embeddings, rms_weights=rms_weights)))


if __name__ == '__main__':
    unittest.main()
