# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import sys

sys.path.append('../../lookahead')
import unittest
import numpy as np

from common.lookahead_cache import Tree


class TestsCache(unittest.TestCase):

    def test_get_one_branch(self):
        tree = Tree(1)
        tree.put([1, 2, 3, 4], mode='output', idx=-1)
        ids, mask, sizes = tree.get([1], max_size=63, max_length=8, min_input_size=0, min_output_size=0,
                                    output_weight=1e-4, mode='mix', idx=0)
        self.assertTrue(len(ids) == 4, ids)

        self.assertTrue(ids == [1, 2, 3, 4], ids)

        self.assertTrue(mask.shape[0] == 4, mask.shape)
        self.assertTrue(mask.shape[1] == 4, mask.shape)

        ref_mask = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertTrue(np.sum(np.abs(mask - ref_mask)) == 0.0, mask)

    def test_get_multi_branches(self):
        tree = Tree(1)
        tree.put([1, 2, 3], mode='output', idx=-1)
        tree.put([1, 2, 4], mode='output', idx=-1)
        ids, mask, sizes = tree.get([1], max_size=63, max_length=8, min_input_size=0, min_output_size=0,
                                    output_weight=1e-4, mode='mix', idx=0)
        self.assertTrue(len(ids) == 4, ids)

        self.assertTrue(ids == [1, 2, 3, 4], ids)

        self.assertTrue(mask.shape[0] == 4, mask.shape)
        self.assertTrue(mask.shape[1] == 4, mask.shape)

        ref_mask = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]])
        self.assertTrue(np.sum(np.abs(mask - ref_mask)) == 0.0, mask)


if __name__ == '__main__':
    unittest.main()
