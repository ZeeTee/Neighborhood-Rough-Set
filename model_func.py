# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 21:18
# @Author  : zeetng

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from datetime import datetime, date, time

from preprocessing import get_partial_labeled_data, trans_to_d, get_data, semi_labels_trans, ReadData, \
    split_unlabel_data, DATASET_NUME_ATTRS
from neighbor_rough_set import NeighborhoodRoughSet
from utils import map_to_dps, group_by_label, get_log_file


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
        return NeighborhoodRoughSet(labeled_data, list(red.union({a})), delta).dependency_to_b(D) \
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
        return imp(A.union({a})) - imp(A)

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
        def __init__(self, data, attrs, nume_attrs, labels: dict, delta):
            super(NeighborSemiRoughSet, self).__init__(data, attrs, nume_attrs, delta)
            self.labels = labels
            self._l1 = set()  # 正域
            self._l2 = set()  # 边界域
            self._l3 = set()  # 不确定
            self.label_name = [x for x in range(len(labels) - 1)]
            self.grouped_sample()

        def grouped_sample(self):
            for i in range(self.n):
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
            return a * (len(self._l1) / self.n) + b * (self._n / (len(self._l2) + self._n)) \
                   + c * (nc / self.compute_granular_card())

    def sig(a, A: set):
        sig_u = NeighborSemiRoughSet(data, list(A.union({a})), D, delta).imp(alpha, beta, gamma, nc)
        sig_a = NeighborSemiRoughSet(data, list(A), D, delta).imp(alpha, beta, gamma, nc)
        return sig_u, sig_a, sig_u - sig_a

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


def my_model(neighbor_delta, labeled_data, unlabeled_data, labels, name, alpha, f):
    """

    :param f: logout file
    :param alpha: radio of labeled data
    :param name: name of dataset
    :param neighbor_delta: float, 邻域半径
    :param labeled_data: ndarray
    :param unlabeled_data: ndarray
    :param labels: ndarray
    :return:
    """
    # 参数
    # data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv'
    delta = neighbor_delta
    beta = 1 - alpha
    # radio = radio

    # 模型
    class SemiRoughSet(NeighborhoodRoughSet):
        def __init__(self, data, attrs, nume_attrs, delta, labels=None):
            """
            :param data: 数据
            :param attrs: 属性列表 list
            :param delta: 邻域半径
            :param labels: 标签
            """
            # 调用父类，建立邻域
            super(SemiRoughSet, self).__init__(data, attrs, nume_attrs, delta)
            if labels is None:
                self.labels = []
            else:
                self.labels = labels
                self.partition = group_by_label(self.labels)

        def __neighbor_upper_app(self, k):
            """
            计算样本k的邻域上近似集
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
            return self.l_dis() / (pow(self.n, 2) - self.n) if len(self.labels) else self.u_dis() / (
                        pow(self.n, 2) - self.n)

    def measure(attrs):
        if len(attrs) == 0:
            return 0
        # 计算数值属性集
        n_attrs = nume_attrs.intersection(attrs)
        model_un = SemiRoughSet(unlabeled_data, list(attrs), list(n_attrs), delta)
        model_la = SemiRoughSet(labeled_data, list(attrs), list(n_attrs), delta, labels)
        ds = beta * model_un.compute_dis() + alpha * model_la.compute_dis()
        return ds

    def imp(a, t: set):
        m_ta = measure(t.union({a}))
        m_t = measure(t)
        diff = m_ta - m_t
        return m_ta, m_t, diff

    # --------------------------算法流程---------------------------
    print("初始化数据......")
    labeled_data = pd.DataFrame(labeled_data)
    unlabeled_data = pd.DataFrame(unlabeled_data)

    # 属性集
    attr = list(labeled_data.columns)  # 条件属性集
    nume_attrs = DATASET_NUME_ATTRS[name]  # 获取数值属性
    # D = trans_to_d(labels.reshape(-1))  # 将决策属性转化格式
    f.write("{}>> 算法开始运行\n".format(datetime.now()))
    f.write("初始条件属性集: {}\n".format(attr))
    f.write("数值属性集：{}\n".format(nume_attrs))

    # Algorithm
    red = set()  # 约简集
    mx_ds_red = 0  # 最大值
    attr = set(attr)
    dsc = measure(attr)
    f.write("所有数据的measure值: {}\n".format(dsc))

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


def new_model(neighbor_delta, labeled_data, unlabeled_data, labels, name, alpha, f):
    """
    :param f: logout file
    :param alpha: radio of labeled data
    :param name: name of dataset
    :param neighbor_delta: float, 邻域半径
    :param labeled_data: ndarray
    :param unlabeled_data: ndarray
    :param labels: ndarray
    :return:
    """
    # 参数
    # data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv'
    delta = neighbor_delta
    beta = 1 - alpha
    # radio = radio

    # 模型
    class SemiRoughSet(NeighborhoodRoughSet):
        def __init__(self, data, attrs, nume_attrs, delta, labels=None):
            """
            :param data: 数据
            :param attrs: 属性列表 list
            :param delta: 邻域半径
            :param labels: 标签
            """
            # 调用父类，建立邻域
            super(SemiRoughSet, self).__init__(data, attrs, nume_attrs, delta)
            if labels is None:
                self.labels = []
            else:
                self.labels = labels
                self.partition = group_by_label(self.labels)

    def measure(attrs):
        if len(attrs) == 0:
            return 0
        # 计算数值属性集
        n_attrs = nume_attrs.intersection(attrs)
        model_un = SemiRoughSet(unlabeled_data, list(attrs), list(n_attrs), delta)
        model_la = SemiRoughSet(labeled_data, list(attrs), list(n_attrs), delta, labels)
        D = trans_to_d(labels.reshape(-1))  # 将决策属性转化格式
        low = model_un.get_lower_approximation(D)
        struct1 = sum(len(neighbor) for neighbor in model_un.group_grans.values())
        struct2 = sum(len(neighbor) for neighbor in model_la.group_grans.values())
        return 0.8 * len(low) / model_la.n + 0.2 * math.exp(- struct1 / pow(model_la.n, 2)) + 0.2 * math.exp(- struct2 / pow(model_un.n, 2))

    def imp(a, t: set):
        m_ta = measure(t.union({a}))
        m_t = measure(t)
        diff = m_ta - m_t
        return m_ta, m_t, diff

    # --------------------------算法流程---------------------------
    print("初始化数据......")
    labeled_data = pd.DataFrame(labeled_data)
    unlabeled_data = pd.DataFrame(unlabeled_data)

    # 属性集
    attr = list(labeled_data.columns)  # 条件属性集
    nume_attrs = DATASET_NUME_ATTRS[name]  # 获取数值属性
    # D = trans_to_d(labels.reshape(-1))  # 将决策属性转化格式
    f.write("{}>> 算法开始运行\n".format(datetime.now()))
    f.write("初始条件属性集: {}\n".format(attr))
    f.write("数值属性集：{}\n".format(nume_attrs))

    # Algorithm
    red = set()  # 约简集
    mx_sig_red = 0  # 最大值
    attr = set(attr)
    # dsc = measure(attr)
    # f.write("所有数据的measure值: {}\n".format(dsc))

    print("开始约简......")
    while True:
        mx_sig = 0
        mx_sig_union = 0
        i = -1
        for a in attr.difference(red):
            sig_union, sig_red, sig_a = imp(a, red)
            print("计算属性{}重要度为：{}".format(a, sig_a))
            if sig_a > mx_sig:
                mx_sig = sig_a
                mx_sig_union = sig_union
                mx_sig_red = sig_red
                i = a
        if i == -1:
            print("本轮运行结果：{}".format(red))
            print("当前约简集重要度：{}".format(mx_sig_red))
            break
        print("当前最重要属性{}".format(i))
        red.add(i)
        print("本轮运行结果：{}".format(red))
        print("当前约简集重要度：{}".format(mx_sig_union))
        print("-----------------------------------------------------")

    return red, mx_sig_red


def run(root, file):
    # 超参数
    # data_path = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/ecoli.csv'
    data_path = root + file + '.csv'
    radio = 0.3
    k_fold = 10
    valid_size = 0.11
    alpha = 0.8
    beta = 0.2
    res = []

    reader = ReadData(data_path, norm=True)
    # 保存运行信息的文件
    f = get_log_file(file)

    for r in [0.5]:  # 未标签率
        for d in np.arange(0.06, 0.20, 0.02):  # w

            red_size = 0
            acc_t = 0.0
            dis_m = 0.0
            print("--------------------------------------------")
            f.write("--------------------------------------------\n")
            f.write("数据集: {}, 当前未标签率: {}, 邻域半径: {}\n".format(file, r, d))
            print("{}==>当前未标签率: {}, 邻域半径: {}".format(file, r, d))
            # 先分割数据集
            for i, x_train, x_test, y_train, y_test in reader.get_k_fold(k_fold):
                f.write("{}th fold:\n".format(i+1))
                # 分割有无标签的数据集
                x_labeled, x_unlabeled, y_labeled, _ = split_unlabel_data(x_train, y_train, r)

                start_time = datetime.now()
                red, ds = new_model(d, x_labeled, x_unlabeled, y_labeled, file, alpha, f)
                end_time = datetime.now()
                red_size += len(red)
                dis_m += ds
                print("{}==>第{}折约简结果: {}, ds: {}".format(file, i, list(red), ds))
                f.write("reduction: {}\nds: {}\nrunning time: {}s\n".format(list(red), ds, (end_time-start_time).seconds))

                # Train SVM with raw data
                clf = SVC()
                clf.fit(x_labeled, y_labeled.ravel())
                acc_test = clf.score(x_test, y_test.ravel())
                f.write("accuracy of SVM with raw data: {}\n".format(acc_test))

                # 根据约简结果处理数据
                reduction = list(red)
                x_labeled, x_test = x_labeled[:, reduction], x_test[:, reduction]

                # Train SVM with reduced data
                clf = SVC()
                clf.fit(x_labeled, y_labeled.ravel())

                acc_test = clf.score(x_test, y_test.ravel())
                acc_t += acc_test

                print("{}==>{}: 第{}折SVM分类结果test: {}".format(file, d, i, acc_test))
                print("--------------------------------------------")

                f.write("accuracy of SVM with reduced data: {}\n".format(acc_test))
                f.write("--------------------------------------------------------\n")

            avg_red_size = red_size / k_fold
            avg_ds = dis_m / k_fold
            avg_acc_test = acc_t / k_fold

            print("约简长度: {}, 准确率: {}".format(avg_red_size, avg_acc_test))
            res.append([alpha, r, d, avg_red_size, avg_ds, avg_acc_test])

    df = pd.DataFrame(res, columns=['alpha', 'unlabeled-radio', 'radius', 'reduction-size', 'ds', 'test-accuracy'])
    df.to_excel("results/results-of-diff-radius/{}-results-of-diff-radius.xlsx".format(file))
    f.close()
    print("{}运行结束".format(file))


if __name__ == '__main__':
    files = ['wine',
             'wdbc',
             'ionosphere',
             # 'mushroom',
             # 'lymphography',
             # 'zoo',
             # 'german',
             # 'heart',
             # 'credit'
             ]
    root = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/'
    for file in files:
        run(root, file)
