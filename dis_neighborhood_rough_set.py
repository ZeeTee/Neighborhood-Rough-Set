# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 21:28
# @Author  : zeetng
import numpy as np
from utils import map_to_dps


class DP_NRS:
    def __init__(self, labeled_data, labels, unlabeled_data, delta):
        self.delta = delta
        # 输入数据
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

        # 数据维度
        self._ln, self._c = labeled_data.shape
        self._un, self._c = unlabeled_data.shape
        # assert self._lc == self._un

        self.labels = np.asarray(labels).reshape(-1, 1)
        assert self._ln == len(labels)
        # 全体属性DP集
        self.u_dpc = self.calculate_dp([], 'unlabeled', C=True)
        self.l_dpc = self.calculate_dp([], 'labeled', C=True)
        #
        self.len_u_dpc = len(self.u_dpc)
        self.len_l_dpc = len(self.l_dpc)

    def __un_label_dp(self, u_data):
        _udp = set()
        if len(u_data) == 0 :
            return _udp
        for i in u_data:
            bool_dis = np.any(u_data - i > self.delta, axis=1)
            _out = np.where(bool_dis)[0]
            _in = np.where(~bool_dis)[0]
            _udp = _udp.union(map_to_dps(_in, _out))
        return _udp

    def __label_dp(self, l_data, labels):
        _ldp = set()
        if len(l_data) == 0:
            return _ldp
        for i, label in zip(l_data, labels):
            bool_dis_attr = np.any(l_data-i > self.delta, axis=1)
            bool_dis_label = np.any(labels != label, axis=1)
            stacked_bool = np.all(np.stack((bool_dis_attr, bool_dis_label)).T, axis=1)
            _out = np.where(stacked_bool)[0]
            _in = np.where(~stacked_bool)[0]
            _ldp = _ldp.union(map_to_dps(_in, _out))
        return _ldp

    def calculate_dp(self, attrs, datatype, C=False):
        """
        计算差别集
        :param attrs: list 计算使用的属性集
        :param datatype: 指定计算的数据类型，可选'labeled'和'unlabeled'
        :param C: bool 指定计算的属性集是否为全体属性集C,当C为True时attrs为空
        :return: None
        """
        if datatype == 'labeled':
            data = self.labeled_data.values if C else self.labeled_data[attrs].values
            dp = self.__label_dp(data, self.labels)
        else:
            data = self.unlabeled_data.values if C else self.unlabeled_data[attrs].values
            dp = self.__un_label_dp(data)
        return dp

    def calculate_IA(self, A):
        if not len(A):
            return 0
        dp_l = self.calculate_dp(A, datatype='labeled')
        dp_u = self.calculate_dp(A, datatype='unlabeled')
        l = len(dp_l.symmetric_difference(self.l_dpc))/(self.len_l_dpc + len(dp_l))
        u = len(dp_u.symmetric_difference(self.u_dpc))/(self.len_u_dpc + len(dp_u))
        return 0.8 * (1-l) + 0.2 * (1-u)

    def run_reduction(self, no_iter):
        print("初始化......")
        p = set()
        c = set(range(self._c))
        i_best = self.calculate_IA(list(c))
        i_cur = 0

        no = 0
        while i_cur < i_best and no < no_iter:
            t = p.copy()
            iter_s = c.difference(p)
            i_t = self.calculate_IA(list(t))
            max_ = 0
            max_i = 0
            for a in iter_s:
                t.add(a)
                i_with_a = self.calculate_IA(list(t))
                diff = i_with_a - i_t
                if diff > max_ and i_with_a > max_i:
                    max_ = a
                    max_i = i_with_a
                t.remove(a)
            p = p.union({max_})
            i_cur = max_i

            print("Reduction set: ", p)
            print("Current I: ", i_cur)
            no += 1
        return p

    def run_backward(self):
        print("初始化......")
        p = set()
        i_best = 1
        i_cur = 0
        c = set(range(self._c))
        no = 0
        while i_cur < i_best:
            t = c.copy()
            iter_s = c.difference(p)
            i_t = self.calculate_IA(list(t))
            max_ = 0
            max_i = 2
            for a in iter_s:
                t.remove(a)
                i_without_a = self.calculate_IA(list(t))
                diff = i_t - i_without_a
                if 0 < diff < max_i:
                    max_ = a
                    max_i = diff
                t.add(a)
            p = p.union({max_})
            i_cur = max_i

            print("Reduction set: ", p)
            print("Current I: ", i_cur)
            no += 1


