# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 21:18
# @Author  : zeetng

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from preprocessing import get_partial_labeled_data, trans_to_d, get_data
from neighbor_rough_set import NeighborhoodRoughSet


def dis_matrix():
    # 读取数据
    raw_data = pd.read_csv('/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv', header=None)
    # 每类选取10个数据，共计30个
    sel_data = raw_data.iloc[0: 10].append(raw_data.iloc[59: 69]).append(raw_data.iloc[130: 140])
    # 为数据重新编行号
    sel_data.reset_index(drop=True, inplace=True)
    # 取出标签
    labels = sel_data.pop(0)

    # 标准化
    sca = MinMaxScaler()
    sca_data = sca.fit_transform(sel_data)
    sca_data = pd.DataFrame(sca_data)

    # sca_data["label"] = labels.values
    # sns.set(style="ticks")
    # sns.pairplot(sca_data, hue="label")
    # plt.show()

    print('test')


def feature_selection_neighborhood_rs():
    # 参数
    delta = 0.11
    data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv'
    # ----------------------------------------------------------------------

    def sig(a, red: set):
        return NeighborhoodRoughSet(labeled_data, list(red.union({a})), delta).dependency_to_b(D)\
               - NeighborhoodRoughSet(labeled_data, list(red), delta).dependency_to_b(D)
    # 读取数据
    print("读取数据......")
    labeled_data, labels = get_data(data_path)

    print("初始化......")
    # 条件属性集
    attr = set(labeled_data.columns)
    # 将决策属性转化格式
    D = trans_to_d(labels.values.reshape(-1))

    red = set()

    print("开始约简......")
    while True:
        mx = 0
        i = -1
        for a in attr.difference(red):
            sa = sig(int(a), red)
            print("计算属性{}重要度为：{}".format(int(a), sa))
            if sa > mx:
                mx = sa
                i = a
        print("当前最重要属性{}".format(i))
        if mx > 0:
            red.add(i)
        else:
            break
        print("本轮运行结果：{}".format(red))
        # print("当前约简集重要度：{}".format(imp_A))
        print("-----------------------------------------------------")
    return red, len(red)


def semi_neighbor_rs():
    # 参数
    data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv'
    radio = 0.7
    delta = 0.1
    alpha = 0.8
    beta = 0.2
    # ----------------------------------------------------------------------
    # 半监督邻域粗糙集模型

    class SemiNeighborRoughSet(NeighborhoodRoughSet):
        def compute_granular_card(self):
            cnt = 0
            for k, v in self.group_grans.items():
                cnt += len(v)
            return cnt

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
            self._n = n
            self._c = c

            if len(attrs) > c:
                raise ValueError("无效的属性列表", attrs)
            else:
                data = data[attrs].values

            def __euclidean_dist(_x, matrix):
                return np.sqrt(np.sum(np.square(_x - matrix), axis=1))

            # 使用了新的邻域半径计算方法
            for sample_i in range(n):
                dist = __euclidean_dist(data[sample_i, :], data)
                dist = np.delete(dist, sample_i, 0)  # 排除样本自己
                radius = min(dist) + delta * (max(dist) - min(dist))
                ed = dist <= radius
                ed = np.insert(ed, sample_i, True)
                _id = np.arange(n)[ed]
                self.group_grans[sample_i] = set(_id)

    def imp(A):
        if not len(A):
            return 0
        gama = SemiNeighborRoughSet(labeled_data, list(A), delta).dependency_to_b(D)
        det = SemiNeighborRoughSet(unlabeled, list(A), delta).compute_granular_card()
        return alpha * (gama / atl) + beta * (atul / det)

    def sig(a, A: set):
        return imp(A.union({a}))-imp(A)

    # ----------------------------------------------------------------------
    # 读取数据
    print("读取数据......")
    labeled_data, labels, unlabeled = get_partial_labeled_data(path=data_path, ratio=radio)

    print("初始化......")

    attr = set(labeled_data.columns)  # 条件属性集

    D = trans_to_d(labels.reshape(-1))  # 将决策属性转化格式

    # Algorithm Compute IMP-Reduct
    atl = SemiNeighborRoughSet(labeled_data, list(attr), delta).dependency_to_b(D)
    atul = SemiNeighborRoughSet(unlabeled, list(attr), delta).compute_granular_card()

    imp_at = 1
    A = set()

    print("开始约简......")
    while True:
        mx = 0
        i = -1
        for a in attr.difference(A):
            sa = sig(int(a), A)
            print("计算属性{}重要度为：{}".format(int(a), sa))
            if sa > mx:
                mx = sa
                i = a
        print("当前最重要属性{}".format(i))
        A.add(i)
        print("本轮运行结果：{}".format(A))
        imp_A = imp(A)
        print("当前约简集重要度：{}".format(imp_A))
        print("-----------------------------------------------------")
        if imp_A >= imp_at:
            break
    return A


if __name__ == '__main__':
    print(semi_neighbor_rs())
