# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 20:43
# @Author  : zeetng
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import logging


class NeighborhoodRoughSet:
    """
    建立邻域粗糙集模型
    """
    def __init__(self,  data: pd.DataFrame, attrs: [], delta=0.2):
        # 邻域半径
        self._delta = delta
        # 样本个数
        self._n = 0
        # 条件属性数
        self._c = 0
        self._attrs = attrs
        # 样本
        self.data = data
        # 邻域粒子族(邻域粒化空间)
        self.group_grans = defaultdict(set)
        # 粒化
        self.granulate(data, attrs, self._delta)
        #
        self.isLASCalculate = False
        self.__lower_approximation_set = set()
        self.isUASCalculate = False
        self.__upper_approximation_set = set()

    def granulate(self, data: pd.DataFrame, attrs: [], delta):
        """
        粒化操作，根据给定的数据和属性索引建立每个样本的邻域粒子
        并保存在group_grans中。每个邻域粒子都是一个集合，其中
        保存邻域样本的id, 默认使用样本索引作为每个样本的id。

        :param delta: 邻域半径
        :param data: DataFrame类型的数据
        :param attrs: 属性的索引
        :return:
        """
        if data is None:
            raise ValueError("无效的输入数据", data)

        n, c = data.shape
        self._delta = delta
        self._n = n
        self._c = c

        if len(attrs) > c:
            raise ValueError("无效的属性列表", attrs)
        else:
            data = data[attrs].values

        # 距离函数
        def __euclidean_dist(_x, matrix):
            return np.sqrt(np.sum(np.square(_x-matrix), axis=1))

        for sample_i in range(n):
            # 传统方法
            ed = __euclidean_dist(data[sample_i, :], data) <= self._delta
            # 新方法
            # dist = __euclidean_dist(data[sample_i, :], data)
            # dist = np.delete(dist, sample_i, 0)  # 排除样本自己
            # radius = min(dist) + delta * (max(dist) - min(dist))
            # ed = dist <= radius
            # ed = np.insert(ed, sample_i, True)
            _id = np.arange(n)[ed]
            self.group_grans[sample_i] = set(_id)

    def compute_granular_card(self):
        cnt = 0
        for k, v in self.group_grans.items():
            cnt += len(v)
        return cnt

    def _lower_approximate(self, x: set):
        """
        计算x的下近似集
        """
        if self.group_grans is not None:
            nx = set()
            for key, value in self.group_grans.items():
                if value.issubset(x):
                    nx.add(key)
            return nx
        else:
            return None

    def _upper_approximate(self, x: set):
        """
        计算x的上近似集
        """
        if self.group_grans is not None:
            nx = set()
            for key, value in self.group_grans.items():
                if value.intersection(x) is not None:
                    nx.add(key)
            return nx
        else:
            return None

    def get_lower_approximation(self, X: dict):
        """
        获取当前邻域空间下，X的下近似集

        :param X: dict {类别1: {样本id...}, 类别2: {样本id...}... }
        :return:
        """
        if not X:
            raise ValueError("输入错误")
        if not self.group_grans:
            raise ValueError("未建立邻域粒空间")
        las = set()
        for value in X.values():
            s = self._lower_approximate(value)
            las = las.union(s)
        self.__lower_approximation_set = las
        return las

    def get_upper_approximation(self, X: dict):
        """
        获取当前邻域空间下，X的上近似集

        :param X: dict {类别1: {样本id...}, 类别2: {样本id...}... }
        :return:
        """
        if not X:
            raise ValueError("输入错误")
        if not self.group_grans:
            raise ValueError("未建立邻域粒空间")
        uas = set()
        for value in X.values():
            s = self._upper_approximate(value)
            uas = uas.union(s)
        self.__upper_approximation_set = uas
        return uas

    def get_boundary(self, X: dict):
        """
        获取当前邻域空间下，X的边界集
        """
        return self.get_upper_approximation(X).difference(self.get_lower_approximation(X))

    def dependency_to_b(self, D: dict):
        """
        计算决策属性D对条件属性B的依赖度
        """
        if not D:
            raise ValueError("输入错误")
        if not self.group_grans:
            raise ValueError("未建立邻域粒空间")
        if not self.isLASCalculate:
            _ = self.get_lower_approximation(D)
        if not self.isUASCalculate:
            _ = self.get_upper_approximation(D)

        logging.info("当前属性集:{}，当前下近似集:{}".format(self._attrs, self.__lower_approximation_set))
        gama = len(self.__lower_approximation_set) / self._n
        return gama


######################################################################
"""
这部分是用于邻域粗糙集的功能函数
"""


def calculate_sig_a(data, delta, a, B: set, D: dict):
    """
    计算属性a的重要度
    :param data: 数据
    :param delta: 邻域半径
    :param a: 属性a
    :param B: 属性集B
    :param D: 决策属性集
    :return: 重要度
    """
    ab_nrs = NeighborhoodRoughSet(data, list(B.union({a})), delta)
    b_nrs = NeighborhoodRoughSet(data, list(B), delta)
    sig_a = ab_nrs.dependency_to_b(D) - b_nrs.dependency_to_b(D)
    return sig_a


def trans_to_d(labels):
    """
    将标签列表转化为{类别：样本集}的格式
    :param labels:
    :return:
    """
    D = defaultdict(set)
    for index, label in enumerate(labels):
        D[label].add(index)
    return D


######################################################################


if __name__ == '__main__':
    # 日志设置
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)

    # prepare data
    # data = pd.read_csv('/Volumes/Seagate Backup Plus Drive/UCI/wdbc/wdbc.csv',
    #                   header=None)
    # label = data[34].copy()
    # data = data.drop(34, axis=1)

    # 数据
    data = pd.DataFrame(
        {0: [0.0909, 0, 0.4091, 0.6364, 1.0000, 0.9091, 0.9545, 0.6818],
         1: [1.0000, 0.3750, 0, 0.7500, 0.3750, 0.5000, 0.6250, 0.6250],
         2: [1, 1, 1, 1, 2, 2, 2, 1],
         3: ['Setosa', 'Setosa', 'Virginica', 'Virginica', 'Virginica', 'Versicolor', 'Versicolor', 'Versicolor']}
    )
    label = data[3].copy()
    # data = data.drop([0, 1], axis=1)
    # data = pd.DataFrame(data.values, columns=[x for x in range(30)])

    # 标准化
    # sca = MinMaxScaler()
    # sca_data = sca.fit_transform(data)
    # sca_data = pd.DataFrame(sca_data)

    # 标签编码
    enc = OrdinalEncoder()
    label = enc.fit_transform(label.values.reshape(-1, 1))

    # 将D转化为{类别：样本集} => {类别1: {样本id...}, 类别2: {样本id...}...}
    D = trans_to_d(label.reshape(-1))

    red = set()  # 约间集
    A = set([x for x in range(3)])  # 属性集A

    while A:
        max_sig = -1
        max_sig_index = -1
        logging.info("开始循环：")
        for a in A.difference(red):
            logging.info("计算属性{}的重要度".format(a))
            sig = calculate_sig_a(data, 0.1930, a, red, D)
            logging.info("属性{}的重要度为{}".format(a, sig))
            if sig > max_sig:
                max_sig = sig
                max_sig_index = a
        if max_sig > 0:
            red = red.union({max_sig_index})
            logging.info("当前最重要的属性是{},{}".format(max_sig_index, max_sig))
        else:
            break

    print(red)

