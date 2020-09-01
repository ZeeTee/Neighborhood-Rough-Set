# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 21:18
# @Author  : zeetng

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from preprocessing import get_partial_labeled_data, trans_to_d, get_data, semi_labels_trans, ReadData, split_unlabel_data
from neighbor_rough_set import NeighborhoodRoughSet
from utils import map_to_dps, group_by_label


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
        pass

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

    D = trans_to_d(labels.values.reshape(-1))  # 将决策属性转化格式

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


def proposed_model():
    # 参数
    data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv'
    radio = 0.3
    delta = 0.1

    alpha = 0.8
    beta = 0.1
    gamma = 0.1
    # ----------------------------------------------------------------------
    # 半监督邻域粗糙集模型

    class NeighborSemiRoughSet(NeighborhoodRoughSet):
        def __init__(self, data, attrs, labels: dict, delta):
            super(NeighborSemiRoughSet, self).__init__(data, attrs, delta)
            self.labels = labels
            self._l1 = set()  # 正域
            self._l2 = set()  # 边界域
            self._l3 = set()  # 不确定
            self.label_name = [x for x in range(len(labels)-1)]
            self.grouped_sample()

        def grouped_sample(self):
            for i in range(self._n):
                # 是否标记数据属于同一类
                cnt = 0  # 统计邻域内有多少种类别的样本
                ll = []  # 保存类别名
                for name in self.label_name:
                    if self.group_grans[i].isdisjoint(self.labels[name]):
                        cnt += 1
                        ll.append(name)
                    if cnt >= 2:
                        self._l2.add(i)  # 属于边界域
                        break
                if cnt == 1:
                    # 当邻域内样本只有一个类别时，考虑邻域内是否有无标签数据，若没有则判定属于正域，
                    # 若有则继续判断数量
                    # 是否拥有无标签数据
                    if self.group_grans[i].isdisjoint(self.labels["unlabeled"]):
                        self._l3.add(i)  # 属于不确定域
                    else:
                        self._l1.add(i)  # 属于正域

        def imp(self, a, b, c, nc):
            return a * (len(self._l1) / self._n) + b * (self._n / (len(self._l2) + self._n)) \
                   + c * (nc / self.compute_granular_card())

    def sig(a, A: set):
        sig_u = NeighborSemiRoughSet(data, list(A.union({a})), D, delta).imp(alpha, beta, gamma, nc)
        sig_a = NeighborSemiRoughSet(data, list(A), D, delta).imp(alpha, beta, gamma, nc)
        return sig_u, sig_a, sig_u-sig_a

    # 读取数据
    print("读取数据......")
    data, labels, unlabeled = get_partial_labeled_data(path=data_path, ratio=radio, only_index=True)

    print("初始化......")

    attr = set(data.columns)  # 条件属性集

    D = semi_labels_trans(labels, unlabeled)  # 将决策属性转化格式

    # base
    base = NeighborSemiRoughSet(data, list(attr), D, delta)
    nc = base.compute_granular_card()
    imp_at = base.imp(alpha, beta, gamma, nc)

    A = set()

    print("开始约简......")
    while True:
        mx = 0
        mx_aa = 0
        i = -1
        for a in attr.difference(A):
            s_aa, s_a, imp = sig(int(a), A)
            print("计算属性{}重要度为：{}".format(int(a), imp))
            if imp > mx:
                mx = imp
                mx_aa = s_aa
                i = a
        print("当前最重要属性{}".format(i))
        A.add(i)
        print("本轮运行结果：{}".format(A))
        print("当前约简集重要度：{}".format(mx_aa))
        print("-----------------------------------------------------")
        if mx_aa >= imp_at:
            break
    return A


def my_model(neighbor_delta, labeled_data, unlabeled_data, labels, alpha, beta):
    """

    :param neighbor_delta: float, 邻域半径
    :param labeled_data: ndarray
    :param unlabeled_data: ndarray
    :param labels: ndarray
    :return:
    """
    # 参数
    # data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv'
    delta = neighbor_delta
    # radio = radio

    # 模型
    class SemiRoughSet(NeighborhoodRoughSet):
        def __init__(self, data, attrs, delta, labels=None):
            """

            :param data: 数据
            :param attrs: 属性列表 list
            :param delta: 邻域半径
            :param labels: 标签
            """
            # 调用父类，建立邻域
            super(SemiRoughSet, self).__init__(data, attrs, delta)
            if labels is None:
                self.labels = []
            else:
                self.labels = labels
                self.partition = group_by_label(self.labels)

        def __neighbor_upper_app(self, k):
            """
            计算样本k的领域上近似集
            :param k:
            :return:
            """
            upper_app = set()
            neibors = self.group_grans[k]
            for n in neibors:
                upper_app = upper_app.union(self.group_grans[n])
            return upper_app

        def __diff_samples(self, k):
            """
            计算样本k邻域内同类的样本数量
            :param k: index
            :return:
            """
            ds = self.group_grans[k].union(self.partition[self.labels[k][0]])
            return ds

        def u_dis(self):
            """
            计算无标签数据的差别计数
            :return:
            """
            cnt = 0
            if len(self.data) == 0:
                return cnt
            for key in self.group_grans.keys():
                cnt += self.n - len(self.__neighbor_upper_app(key))
            return cnt

        def l_dis(self):
            """
            计算有标签数据的差别计数
            :return:
            """
            cnt = 0
            if len(self.data) == 0:
                return cnt
            # 遍历所有样本
            for key in self.group_grans.keys():
                cnt += self.n - len(self.__diff_samples(key))
            return cnt

        def compute_dis(self):
            return self.l_dis() / (pow(self.n, 2) - self.n) if len(self.labels) else self.u_dis() / (pow(self.n, 2)-self.n)

    # 读取数据
    # print("读取数据......")
    # labeled_data, labels, unlabeled = get_partial_labeled_data(path=data_path, ratio=radio)
    # labeled_data, labels = get_data(data_path)

    print("初始化......")
    labeled_data = pd.DataFrame(labeled_data)
    unlabeled_data = pd.DataFrame(unlabeled_data)

    attr = list(labeled_data.columns)  # 条件属性集
    # D = trans_to_d(labels.reshape(-1))  # 将决策属性转化格式

    def measure(attrs):
        if len(attrs) == 0:
            return 0
        model_un = SemiRoughSet(unlabeled_data, list(attrs), delta)
        model_la = SemiRoughSet(labeled_data, list(attrs), delta, labels)
        ds = beta * model_un.compute_dis() + alpha * model_la.compute_dis()
        return ds

    def imp(a, t: set):
        m_ta = measure(t.union({a}))
        m_t = measure(t)
        diff = measure(t.union({a})) - measure(t)
        return m_ta, m_t, diff

    # Algorithm
    red = set()
    mx_ds_red = 0
    attr = set(attr)
    dsc = measure(attr)

    print("开始约简......")
    while True:
        mx_ds = 0
        mx_union = 0
        i = -1
        for a in attr.difference(red):
            ds_union, ds_red, imp_a = imp(a, red)
            print("计算属性{}重要度为：{}".format(a, imp_a))
            if imp_a > mx_ds:
                mx_ds = imp_a
                mx_union = ds_union
                mx_ds_red = ds_red
                i = a
        if i == -1:
            print("本轮运行结果：{}".format(red))
            print("当前约简集重要度：{}".format(mx_ds_red))
            break
        print("当前最重要属性{}".format(1))
        red.add(i)
        print("本轮运行结果：{}".format(red))
        print("当前约简集重要度：{}".format(mx_union))
        print("-----------------------------------------------------")
        if mx_union >= dsc:
            break
    return red, mx_union


if __name__ == '__main__':
    # 超参数
    data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/ecoli.csv'
    radio = 0.3
    k_fold = 10
    valid_size = 0.11
    alpha = 0.8
    beta = 0.2
    res = []

    reader = ReadData(data_path)

    for r in [0.3, 0.5, 0.7]:  # 未标签率
        for d in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:  # w
            red_size = 0
            acc_v = 0.0
            acc_t = 0.0
            dis_m = 0.0
            print("--------------------------------------------")
            print("当前未标签率: {}, 邻域半径: {}".format(r, d))
            for i, x_train, x_valid, x_test, y_train, y_valid, y_test in reader.get_k_fold(k_fold, valid_size):
                # 分割有无标签的数据集
                x_labeled, x_unlabeled, y_labeled, _ = split_unlabel_data(x_train, y_train, r)
                red, ds = my_model(d, x_labeled, x_unlabeled, y_labeled, alpha, beta)
                # red = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                # ds = 1
                red_size += len(red)
                dis_m += ds
                print("第{}折约简结果: {}, ds: {}".format(i, list(red), ds))
                # 根据约简结果重新处理数据
                reduction = list(red)
                x_labeled, x_valid, x_test = x_labeled[:, reduction], x_valid[:, reduction], x_test[:, reduction]
                # SVM 分类
                clf = SVC()
                clf.fit(x_labeled, y_labeled.ravel())

                acc_valid = clf.score(x_valid, y_valid.ravel())
                acc_test = clf.score(x_test, y_test.ravel())
                print("第{}折SVM分类结果valid: {}, test: {}".format(i, acc_valid, acc_test))

                acc_v += acc_valid
                acc_t += acc_test

            avg_red_size = red_size/k_fold
            avg_ds = dis_m/k_fold
            avg_acc_valid = acc_v/k_fold
            avg_acc_test = acc_t/k_fold
            print("约简长度: {}, 准确率: {}".format(avg_red_size, avg_acc_test))
            res.append([r, d, avg_red_size, avg_ds, avg_acc_valid, avg_acc_test])

    df = pd.DataFrame(res, columns=['radio', 'delta', 'reduction-size', 'ds',  'valid-accuracy', 'test-accuracy'])
    df.to_excel("wdbc-results.xlsx")
