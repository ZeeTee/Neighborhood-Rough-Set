# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 16:30
# @Author  : zeetng

# realize rough set based semi-supervised feature selection via ensemble selector

from sklearn.semi_supervised import LabelPropagation
import numpy as np
import pandas as pd
from preprocessing import ReadData, split_unlabel_data, DATASET_NUME_ATTRS
from neighbor_rough_set import NeighborhoodRoughSet, sig_a, trans_to_d
from collections import Counter


class Neighborhood_Classifiers:
    def __init__(self, dataset, train_data, labels, A, delta):
        """
        Realize of Neighborhood classifiers. This is the classifier of RSES to 98
        :param delta:
        :param labels:
        :param train_data:
        :param dataset:
        :return:
        """
        A = list(A)
        nume_attrs = DATASET_NUME_ATTRS[dataset]  # 获取数值属性

        data = pd.DataFrame(train_data)
        self.nrs = NeighborhoodRoughSet(data, A, nume_attrs, delta, dist_func='EUCD')
        self.labels = labels

    def predict(self, test_sample):
        neibors = list(self.nrs.group_grans[test_sample])
        d_neighbors = [self.labels[x] for x in neibors]
        pred = np.argmax(np.bincount(d_neighbors))
        return pred


def compute_NDER(A, X, d):
    """

    :param A: feature set
    :param X: index of samples
    :param d:
    :return:
    """
    if not len(A):
        return 1

    clf = Neighborhood_Classifiers('wine', x, y, A, delta)
    pre = []
    for s in X:
        pre.append(clf.predict(s))
    cnt = np.sum(np.not_equal(pre, d[X]).astype(int), axis=0)
    NDER = cnt / l
    return NDER


if __name__ == '__main__':
    PATH = "/Users/zhang/gitdir/Neighborhood-Rough-Set/dataset/wine.csv"
    K = 10
    delta = 0.2

    reader = ReadData(PATH)
    for r in np.arange(0.1, 1, 0.1):
        for i, x_train, x_test, y_train, y_test in reader.get_k_fold(K):
            # 分割有无标签的数据集
            x_labeled, x_unlabeled, y_labeled, y_unlabeled = split_unlabel_data(x_train, y_train, r)

            # LPA生成标签
            lp_model = LabelPropagation()
            lp_model.fit(x_labeled, np.reshape(y_labeled, len(y_labeled)))
            y_inductive = lp_model.predict(x_unlabeled)

            # 整合数据
            x = np.concatenate((x_labeled, x_unlabeled), axis=0)
            y_labeled = np.reshape(y_labeled, (len(y_labeled)))
            y = np.concatenate((y_labeled, y_inductive), axis=0)

            # Algorithm of rough set based semi-supervised feature selection via
            # ensemble selector.
            # n is number of decision classes
            n = np.max(y_labeled)+1
            l, c = x.shape
            AT = set([x for x in range(c)])
            A = set()

            # compute NDER
            X = [i for i in range(len(x))]
            NDER = compute_NDER(AT, X, y)

            # 按类别分组
            Xi = {i: np.where(y == i)[0].tolist() for i in range(n)}

            while True:
                C = set()
                for i in range(n):
                    max_phi = -1
                    for a in AT.difference(A):
                        l_nder1 = compute_NDER(A, Xi[i], y)
                        A.add(a)
                        l_nder2 = compute_NDER(A, Xi[i], y)
                        phi = l_nder1 - l_nder2
                        if phi > max_phi:
                            max_phi = phi
                            b = a
                        A.remove(a)
                    C.add(b)
                counter = Counter(C)
                b, t = counter.most_common(1)[0]  # 返回n个出现次数最大的值。计数值相等的元素按首次出现的顺序排序。
                A.add(b)
                nder_A = compute_NDER(A, X, y)
                if nder_A <= NDER:
                    break
            print(A)


