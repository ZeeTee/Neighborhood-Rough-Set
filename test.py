# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 21:41
# @Author  : zeetng
import unittest
import numpy as np
import pandas as pd
from neighbor_rough_set import NeighborhoodRoughSet
from dis_neighborhood_rough_set import DP_NRS


class TestRoughSets(unittest.TestCase):
    d = {'a': [0.10, 0.13, 0.14, 0.16],
         'b': [0.20, 0.22, 0.23, 0.41],
         'c': [0.61, 0.56, 0.40, 0.30],
         'd': [0.20, 0.10, 0.31, 0.16]}
    d2 = {0: [0.0909, 0, 0.4091, 0.6364, 1.0000, 0.9091, 0.9545, 0.6818],
          1: [1.0000, 0.3750, 0, 0.7500, 0.3750, 0.5000, 0.6250, 0.6250],
          2: [1, 1, 1, 1, 2, 2, 2, 1],
          'd': [0, 0, 1, 1, 1, 2, 2, 2]}
    data = pd.DataFrame(data=d2)
    nrs = NeighborhoodRoughSet(data, [0, 1,  2], delta=0.1930)

    # print(data[['a1', 'a2']])
    dnrs = DP_NRS(data, data['d'], data[[0, 1, 2]], 0.1930)

    def test_reduction(self):
        print(self.dnrs.run_reduction(100))

    def test_dp(self):
        print(self.dnrs.calculate_dp(['a1'], datatype='labeled', C=False))

    def test_clacu(self):
        print(self.dnrs.calculate_dp(['a1', 'a2']))

    def test_granulate(self):
        self.nrs.granulate(self.data, attrs=['a1'])
        print(self.nrs.group_grans)
        # self.assertDictEqual({0: {0, 1},
        #                       1: {0, 1},
        #                       2: {2},
        #                       3: {3, 7},
        #                       4: {4, 5, 6},
        #                       5: {4, 5, 6},
        #                       6: {4, 5, 6},
        #                       7: {3, 7}}, self.nrs.group_grans)

    def test_get_low_approximation(self):
        self.nrs.granulate(self.data, attrs=['a1', 'a2'])
        nx = self.nrs.get_lower_approximation({1: {0, 1}, 2: {2, 3, 4}, 3: {5, 6, 7}})
        print(nx)

    def test_get_upper_approximation(self):
        self.nrs.granulate(self.data, attrs=['a1', 'a2'])
        nx = self.nrs.get_upper_approximation({1: {0, 1}, 2: {2, 3, 4}, 3: {5, 6, 7}})
        print(nx)
