# -*- coding: utf-8 -*-
# @Time    : 2020/3/23 21:28
# @Author  : zeetng
import numpy as np
import pandas as pd
from utils import map_to_dps


class DP_NRS:
    def __init__(self, labeled_data, labels, unlabeled_data, delta):
        self.delta = delta
        # 输入数据
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

        # 数据维度
        self._ln, self._c = labeled_data.shape
        # self._un, self._c = unlabeled_data.shape
        # assert self._lc == self._un

        self.labels = np.asarray(labels).reshape(-1, 1)
        assert self._ln == len(labels)

        # 全体属性DP集
        self.u_dpc = self.calculate_dp([], 'unlabeled', C=True)
        self.l_dpc = self.calculate_dp([], 'labeled', C=True)
        #
        self.len_u_dpc = len(self.u_dpc)
        self.len_l_dpc = len(self.l_dpc)
        # 保存每个样本的邻域
        self.group_grans = dict()

    def compute_neibors(self, data: pd.DataFrame, attrs: [], delta):
        """
        根据给定的数据和属性索引建立每个样本的邻域粒子,并将结果保存在字典group_grans中。
        其中默认使用样本索引作为每个样本的id。

        :param delta: 邻域半径
        :param data: DataFrame类型的数据
        :param attrs: 属性的索引
        :return:
        """
        if data is None:
            raise ValueError("无效的输入数据", data)

        n, c = data.shape

        if len(attrs) > c:
            raise ValueError("无效的属性列表", attrs)
        else:
            data = data[attrs].values

        # 距离函数
        def __euclidean_dist(_x, matrix):
            return np.sqrt(np.sum(np.square(_x-matrix), axis=1))

        for sample_i in range(n):
            # 传统方法
            # ed = __euclidean_dist(data[sample_i, :], data) <= self._delta
            # 新方法
            dist = __euclidean_dist(data[sample_i, :], data)
            dist = np.delete(dist, sample_i, 0)  # 排除样本自己
            radius = min(dist) + delta * (max(dist) - min(dist))
            ed = dist <= radius
            ed = np.insert(ed, sample_i, True)
            _id = np.arange(n)[ed]
            self.group_grans[sample_i] = set(_id)

    def __dps_func(self, l_data):
        """
        计算邻域差别对， 只考虑属性间距离，不考虑其他因素
        :param l_data:
        :return:
        """
        _ldp = set()
        if len(l_data) == 0:
            return _ldp
        for i in l_data:
            bool_dis_attr = np.any(l_data-i > self.delta, axis=1)
            # bool_dis_label = np.any(labels != label, axis=1)
            # stacked_bool = np.all(bool_dis_attr)
            _out = np.where(bool_dis_attr)[0]
            _in = np.where(~bool_dis_attr)[0]
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
            dp = self.__dps_func(data)
        else:
            if not self.unlabeled_data:
                dp = set()
            else:
                data = self.unlabeled_data.values if C else self.unlabeled_data[attrs].values
                dp = self.__dps_func(data)
        return dp

    def calculate_IA(self, A):
        if not len(A):
            return 0
        dp_l = self.calculate_dp(A, datatype='labeled')
        # dp_u = self.calculate_dp(A, datatype='unlabeled')
        l = len(dp_l.symmetric_difference(self.l_dpc))/(self.len_l_dpc + len(dp_l))
        # u = len(dp_u.symmetric_difference(self.u_dpc))/(self.len_u_dpc + len(dp_u))
        return 1-l

    def run_reduction(self, no_iter, algorithm, threshold):
        if algorithm == "forward":
            print("运行正向约简算法")
            self.run_forward(no_iter)
        else:
            print("运行后向约简算法")
            self.run_backward(threshold)

    def run_forward(self, no_iter):
        print("初始化......")
        p = set()  # 约简集
        c = set(range(self._c))
        i_best = self.calculate_IA(list(c))
        i_cur = 0

        no = 0
        while i_cur < i_best - 0.1 and no < no_iter:
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

    def run_backward(self, threshold):
        print("初始化......")
        p = set()  # 约简结果集合
        i_cur = 1
        c = set(range(self._c))
        no = 0
        t = c.copy()
        while i_cur >= threshold:
            i_t = self.calculate_IA(list(t))  # 当前全部备选属性的I值
            max_diff = 0  # 保存最大的变化值
            max_index = -1  # 保存产生最大变化值的属性索引

            # 计算每个属性去除后对I值的影响，取出最大的
            for a in t:
                t.remove(a)
                i_without_a = self.calculate_IA(list(t))
                diff = i_t - i_without_a
                if diff > max_diff:
                    max_diff = diff
                    max_index = a
                t.add(a)

            # 得到最大的变化属性, 同时保证得到的差异值大于给定的阈值
            # 当得到的最大值已经不满足阈值时，结束约简
            if max_index != -1 and max_diff >= threshold:
                t.remove(max_index)
                p = p.union({max_index})
                i_cur = max_diff
            else:
                break

            print("Reduction set: ", p)
            print("Current I: ", i_cur)
            no += 1


