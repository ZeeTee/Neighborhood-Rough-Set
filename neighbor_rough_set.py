# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 20:43
# @Author  : zeetng
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import numpy.ma as ma
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import logging

from preprocessing import get_data


class NeighborhoodRoughSet:
    """
    建立邻域粗糙集模型
    """
    def __init__(self,
                 data: pd.DataFrame,
                 attrs: [],
                 nume_attrs: [],
                 delta=0.2,
                 delta_type='constant',
                 dist_func='HEOM'):
        """

        :param data:
        :param attrs:
        :param nume_attrs:
        :param delta:
        :param delta_type: 'variable' or 'constant'
        :param dist_func: "HEOM" or "EUCD"
        """
        # 邻域半径
        self.delta_type = delta_type
        self.delta = delta
        # 样本
        self.data = data

        n, c = self.data.shape
        # 样本个数
        self.n = n
        # 条件属性数
        self.c = c
        self.all_attrs = set(attrs)
        self.all_nume_attrs = set(nume_attrs).intersection(self.all_attrs)
        self.all_cate_attrs = self.all_attrs.difference(self.all_nume_attrs)
        self.attrs, self.cate_attrs, self.nume_attrs = \
            list(self.all_attrs), list(self.all_cate_attrs), list(self.all_nume_attrs)

        # 由于保存样本间的距离
        self.dist_func = dist_func
        self.distance = pd.DataFrame([[-1]*n for _ in range(n)])

        # 邻域粒子族(邻域粒化空间)
        self.group_grans = defaultdict(set)
        # 粒化
        self.granulate()

        # 上下近似集
        # self.isLASCalculate = False
        # self.__lower_approximation_set = set()
        # self.isUASCalculate = False
        # self.__upper_approximation_set = set()

    def granulate(self):
        """
        粒化操作，根据给定的数据和属性索引建立每个样本的邻域粒子
        并保存在group_grans中。每个邻域粒子都是一个集合，其中
        保存邻域样本的id, 默认使用样本索引作为每个样本的id。
        """
        if self.data is None:
            raise ValueError("无效的输入数据", self.data)

        if len(self.attrs) > self.c:
            raise ValueError("无效的属性列表", self.attrs)

        # 距离函数
        def __euclidean_dist(_x, matrix, nume_attrs):
            return np.sqrt(np.sum(np.square(_x[nume_attrs]-matrix[:, nume_attrs]), axis=1))

        # 用于计算类别与数值属性混合距离的函数
        def __heom_dist(x: np.ndarray, y: np.ndarray, cates, numes, attrs_max, attrs_min):
            """
            计算HEOM距离
            :param x: 样本x
            :param y: 样本y
            :param numes: 数值属性集
            :param attrs_max: 每个属性的最大值列表
            :param attrs_min: 每个属性的最小值列表
            :return:
            """
            size = len(cates) + len(numes)
            if size == 0:
                return np.zeros(y.shape[0])
            # 计算类别属性的距离
            overlap = np.sum(np.not_equal(x[cates], y[:, cates]).astype(int), axis=1)
            # 计算数值属性的距离
            ab_diff = np.abs(x[numes] - y[:, numes])
            # interval = attrs_max[numes] - attrs_min[numes]
            rn_diff = np.sum(np.square(ab_diff), axis=1)
            dist = np.sqrt(np.add(overlap, rn_diff) / size)
            return dist

        # 计算样本间的距离
        if self.dist_func == "HEOM":
            for si in range(self.n):
                self.distance.loc[si, si:] = self.distance.loc[si:, si] = __heom_dist(
                    self.data.loc[si, :].values,
                    self.data.loc[si:, :].values,
                    self.cate_attrs,
                    self.nume_attrs,
                    self.data.max().values,
                    self.data.min().values)
        elif self.dist_func == "EUCD":
            if len(self.cate_attrs):
                raise TypeError("EUCD方法只能应用于连续型数值属性")

            for si in range(self.n):
                self.distance.loc[si, si:] = self.distance.loc[si:, si] = __euclidean_dist(
                    self.data.loc[si, :].values,
                    self.data.loc[si:, :].values,
                    self.nume_attrs
                )
        else:
            raise TypeError("不合理的参数输入:{}".format(self.dist_func))

        # 当采用数据依赖的邻域半径时，根据数据间距离计算邻域半径
        # 邻域半径"constant"
        self.delta = np.array([self.delta for _ in range(self.n)])
        if self.delta_type == 'variable':
            for i in range(self.n):
                # mask = np.ones(self.distance.shape, dtype=bool)
                # np.fill_diagonal(mask, 0)
                mask = np.ones(self.n, dtype=bool)
                mask[i] = False
                max_dist = self.distance.values[i][mask].max()
                min_dist = self.distance.values[i][mask].min()
                self.delta[i] = min_dist + self.delta[i] * (max_dist - min_dist)

        # 统计邻域的元素
        for si in range(self.n):
            neighbors = np.where(self.distance.values[si, :] < self.delta[si])
            self.group_grans[si] = set(neighbors[0])

    def build(self):
        """
        根据初始化的数据建立邻域粗糙集
        :return:
        """

        pass

    def rebuild(self, attrs):
        """
        根据给定的属性集重建邻域粗糙集模型
        :param attrs:
        :return:
        """
        self.attrs = attrs
        self.nume_attrs = set(self.attrs).intersection(set(self.nume_attrs))
        self.cate_attrs = set(self.attrs).intersection(set(self.cate_attrs))
        self.granulate()

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
        las = self.get_lower_approximation(D)
        # ups = self.get_upper_approximation(D)

        logging.info("当前属性集:{}，当前下近似集:{}".format(self.attrs, las))
        gama = len(las) / self.n
        return gama


######################################################################
"""
这部分是用于邻域粗糙集的功能函数
"""


def sig_a(a, B, D: dict, delta, data, numes):
    """
    计算属性a的重要度
    :param numes:
    :param data: 数据
    :param delta: 邻域半径
    :param a: 属性a
    :param B: 属性集B
    :param D: 决策属性集
    :return: 重要度
    """
    ab_nrs = NeighborhoodRoughSet(data, list(B.union({a})), numes, delta)
    b_nrs = NeighborhoodRoughSet(data, list(B), numes, delta)
    sig = ab_nrs.dependency_to_b(D) - b_nrs.dependency_to_b(D)
    return sig


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


def bool_num_attrs(attrs: list, size: int):
    """
    用于将[0, 2,...,n]形式的数值属性集转换为[True, False, ..., True]的Bool集
    :param attrs:
    :param size:
    :return:
    """
    attrs = set(attrs)
    return [True if x in attrs else False for x in range(size)]


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
    data, label = get_data('dataset/dermatology.csv', False)

    # 数据
    # data = pd.DataFrame(
    #     {0: [0, 0, 0.4091, 0.6364, 1.0000, 0.9091, 0.9545, 0.6818],
    #      1: [1, 0.3750, 0, 0.7500, 0.3750, 0.5000, 0.6250, 0.6250],
    #      2: [1, 1, 1, 1, 2, 2, 2, 1],
    #      3: ['Setosa', 'Setosa', 'Virginica', 'Virginica', 'Virginica', 'Versicolor', 'Versicolor', 'Versicolor']}
    # )
    # label = data[3].copy()
    # data = data.drop([0, 1], axis=1)
    # data = pd.DataFrame(data.values, columns=[x for x in range(30)])

    # 标准化
    # sca = MinMaxScaler()
    # sca_data = sca.fit_transform(data)
    # sca_data = pd.DataFrame(sca_data)

    # 标签编码
    # enc = OrdinalEncoder()
    # label = enc.fit_transform(label.values.reshape(-1, 1))

    # 将D转化为{类别：样本集} => {类别1: {样本id...}, 类别2: {样本id...}...}
    # D = trans_to_d(label.reshape(-1))
    D = dict()
    NeighborhoodRoughSet(data, [x for x in range(34)], [33], 0.5, delta_type='variable')

    red = set()  # 约间集
    A = set([x for x in range(3)])  # 属性集A

    while A:
        max_sig = -1
        max_sig_index = -1
        logging.info("开始循环：")
        for a in A.difference(red):
            logging.info("计算属性{}的重要度".format(a))
            sig = sig_a(data, 0.1930, a, red, D)
            logging.info("属性{}的重要度为{}".format(a, sig))
            if sig > max_sig:
                max_sig = sig
                max_sig_index = a
        if max_sig > 0:
            red = red.union({max_sig_index})
            logging.info("当前最重要的属性是{},{}".format(max_sig_index, max_sig))
        else:
            break

    # C = list(train_data.columns)  # set of condition attributes
    # nume_attrs = DATASET_NUME_ATTRS[dataset]  # 获取数值属性
    # D = trans_to_d(labels)
    # Forward attribute reduction algorithm
    # red = set()
    # C = set(C)
    # max_sig, max_k = 0, -1
    # while len(C) != len(red):
    #     for a in C.difference(red):
    #         siga = sig_a(a, red, D, delta, train_data, nume_attrs)
    #         if siga > max_sig:
    #             max_sig = siga
    #             max_k = a
    #     if max_sig > 0:
    #         red.add(max_k)
    #     else:
    #         return red

    print(red)

