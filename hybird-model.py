# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 15:27
# @Author  : zeetng

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from datetime import datetime, date, time


from preprocessing import get_partial_labeled_data, trans_to_d, get_data, semi_labels_trans, ReadData, \
    split_unlabel_data, DATASET_NUME_ATTRS, hybrid_data
from neighbor_rough_set import NeighborhoodRoughSet
from utils import map_to_dps, group_by_label, get_log_file


class HybridRoughSet(NeighborhoodRoughSet):
    def __init__(self, data, attrs, nume_attrs, delta):
        """

        :param data: hybrid data
        :param attrs:  set of condition attributes
        :param nume_attrs: set of numeric attributes
        :param delta:  threshold of neighborhood

        """
        # 调用父类，建立邻域
        data = pd.DataFrame(data)
        super(HybridRoughSet, self).__init__(data, attrs, nume_attrs, delta)
        self.lapp_p = dict()

    def get_lower_approximation(self, X: dict):
        """

        :param X: key = -1 indicates that sample i is unlabeled data, each key != -1 is the label of sample i
        :return:
        """
        if not X:
            raise ValueError("输入错误")
        if not self.group_grans:
            raise ValueError("未建立邻域粒空间")

        lapp = set()
        for key, values in X.items():
            # 计算Key类的下近似集
            if key == -1:
                continue
            extend = values.union(X[-1])

            # Only samples in extend could belong to lower approximation.
            # If neither the sample nor its neighbors have labels then the sample belong to Uncertainty.
            for i in extend:
                if self.group_grans[i].issubset(extend) and not self.group_grans[i].issubset(X[-1]):
                    lapp.add(i)
                    self.lapp_p[i] = len(self.group_grans[i].intersection(values)) / len(self.group_grans[i])
        return lapp

    def get_upper_approximation(self, X: dict):
        if not X:
            raise ValueError("输入错误")
        if not self.group_grans:
            raise ValueError("未建立邻域粒空间")

        uapp = set()
        for key, values in X.items():
            # 计算Key类的下近似集
            if key == -1:
                continue
            extend = values.union(X[-1])

            for i in range(self.n):
                if self.group_grans[i].intersection(extend) is not None and not self.group_grans[i].issubset(X[-1]):
                    uapp.add(i)
        return uapp

    def gamma(self, X, ):
        lapp = self.get_lower_approximation(X)
        # uapp = self.get_upper_approximation(X)
        return sum(self.lapp_p.values())


def my_model(delta, data, labels, name, alpha, f):
    def sig(a, t):
        s_t = HybridRoughSet(data, list(t), numes, delta).gamma(D)
        t.add(a)
        s_at = HybridRoughSet(data, list(t), numes, delta).gamma(D)
        t.remove(a)
        return s_at, s_t, s_at-s_t
    # --------------------------算法流程---------------------------

    # 属性集
    attrs = [x for x in range(data.shape[1])]  # 条件属性集
    numes = DATASET_NUME_ATTRS[name]  # 获取数值属性集
    D = trans_to_d(labels.reshape(-1))  # 将决策属性转化格式

    f.write("{}>>算法开始运行\n".format(datetime.now()))
    f.write("初始条件属性集: {}\n".format(attrs))
    f.write("数值属性集：{}\n".format(numes))

    # Algorithm
    red = set()  # 约简集
    mx_ds_red = 0  # 最大值
    attrs = set(attrs)

    print("开始约简......")
    while True:
        mx_imp = 0
        mx_union = 0
        i = -1
        for a in attrs.difference(red):
            imp_union, imp_red, imp_a = sig(a, red)
            print("计算属性{}重要度为：{}".format(a, imp_a))
            if imp_a > mx_imp:
                mx_imp = imp_a
                mx_union = imp_union
                mx_red = imp_red
                i = a
        if i == -1:
            print("本轮运行结果：{}".format(red))
            print("当前约简集重要度：{}".format(mx_ds_red))
            break
        print("当前最重要属性{}:{}".format(i, mx_imp))
        if mx_imp <= 0:
            break
        else:
            red.add(i)
        print("本轮运行结果：{}".format(red))
        print("当前约简集重要度：{}".format(mx_union))
    print("-----------------------------------------------------")
    return red, mx_union


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

    reader = ReadData(data_path, norm=False)
    # 保存运行信息的文件
    f = get_log_file(file)

    for r in [0.3, 0.5, 0.7]:  # 未标签率
        for d in np.arange(0.01, 0.31, 0.01):  # w

            red_size = 0
            acc_t = 0.0
            dis_m = 0.0
            print("--------------------------------------------")
            f.write("--------------------------------------------\n")
            f.write("当前未标签率: {}, 邻域半径: {}\n".format(r, d))
            print("当前未标签率: {}, 邻域半径: {}".format(r, d))
            # 先分割数据集
            for i, x_train, x_test, y_train, y_test in reader.get_k_fold(k_fold):
                f.write("{}th fold:\n".format(i+1))
                # 分割有无标签的数据集
                x_train, y_split = hybrid_data(x_train, y_train, r)

                start_time = datetime.now()
                red, ds = my_model(d, x_train, y_split, file, alpha, f)
                end_time = datetime.now()
                red_size += len(red)
                dis_m += ds
                print("第{}折约简结果: {}, ds: {}".format(i, list(red), ds))
                f.write("reduction: {}\nds: {}\nrunning time: {}s\n".format(list(red), ds, (end_time-start_time).seconds))
                print("--------------------------------------------")

                # Train SVM with raw data
                clf = SVC()
                clf.fit(x_train, y_train.ravel())
                acc_test = clf.score(x_test, y_test.ravel())
                f.write("accuracy of SVM with raw data: {}\n".format(acc_test))

                # 根据约简结果处理数据
                reduction = list(red)
                x_labeled, x_test = x_train[:, reduction], x_test[:, reduction]

                # Train SVM with reduced data
                clf = SVC()
                clf.fit(x_labeled, y_train.ravel())

                acc_test = clf.score(x_test, y_test.ravel())
                acc_t += acc_test

                print("第{}折SVM分类结果test: {}".format(i, acc_test))
                f.write("accuracy of SVM with reduced data: {}\n".format(acc_test))
                f.write("--------------------------------------------------------\n")

            avg_red_size = red_size / k_fold
            avg_ds = dis_m / k_fold
            avg_acc_test = acc_t / k_fold

            print("约简长度: {}, 准确率: {}".format(avg_red_size, avg_acc_test))
            res.append([alpha, r, d, avg_red_size, avg_ds, avg_acc_test])

    df = pd.DataFrame(res, columns=['alpha', 'unlabeled-radio', 'radius', 'reduction-size', 'ds', 'test-accuracy'])
    df.to_excel("results/{}-results.xlsx".format(file))
    f.close()
    print("{}运行结束".format(file))


if __name__ == '__main__':
    files = ['wine', 'wdbc', 'ecoli', 'BreastTissue', 'seeds', 'ionosphere', 'parkinsons', 'glass', 'dermatology', 'german', 'yeast', 'segment']
    root = '/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/'
    # for file in files:
    run(root, 'wine')
