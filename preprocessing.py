# -*- coding: utf-8 -*-
# @Time    : 2020/4/13 22:12
# @Author  : zeetng

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# ---------------包含数据预处理的各种操作-------------------------

# scaled_data["label"] = labels.values
# sns.set(style="ticks")
# sns.pairplot(scaled_data, hue="label")
# plt.show()

# 数据集路径
DATA_PATH = 'dataset/wine.csv'

# 数据属性集数值属性
DATASET_NUME_ATTRS = {'wine': set([x for x in range(13)]),
                      'wdbc': set([x for x in range(30)]),
                      'ecoli': set([x for x in range(7)]),
                      'BreastTissue': set([x for x in range(9)]),
                      'seeds': set([x for x in range(7)]),
                      'ionosphere': set([x for x in range(34)]),
                      'parkinsons': set([x for x in range(22)]),
                      'glass': set([x for x in range(9)]),
                      'dermatology': {34},
                      'german': {1, 3, 9},
                      'yeast': set([x for x in range(8)]),
                      'segment': set([x for x in range(19)]),
                      }


def get_data(path=DATA_PATH, norm=True) -> (pd.DataFrame, pd.DataFrame):
    """
    根据给定的路径获取数据集
    :param path: String, path of dataset
    :param norm: Bool, normalize or not
    :return: DataFrame,
    """

    # 读取数据
    raw_data = pd.read_csv(path, header=None)

    # 分离标签
    labels = raw_data.pop(0)
    enc = LabelEncoder()
    enc_labels = enc.fit_transform(labels)

    labels = pd.DataFrame(enc_labels)
    data = pd.DataFrame(raw_data.values)

    # 标准化
    sca = MinMaxScaler()
    sca_data = sca.fit_transform(data)
    scaled_data = pd.DataFrame(sca_data)
    if norm:
        return scaled_data, labels
    else:
        return data, labels


def get_partial_labeled_data(ratio, path=DATA_PATH, norm=True, seed=34, only_index=False):
    """
    获取部分标记数据
    :param only_index: Bool, 是否只返回无标记数据的索引
    :param seed: Int, 随取采样的种子
    :param norm: Bool, 是否归一化
    :param path: String, 读取数据路径
    :param ratio: Float, 未标记数据比例
    :return: DataFrame
    """
    raw, labels = get_data(path, norm)
    # 按比例划分标记与未标记数据
    unlabeled = raw.sample(frac=ratio, random_state=seed)
    label = labels.drop(list(unlabeled.index))
    if not only_index:
        raw = raw.drop(unlabeled.index, axis=0)

    return raw, label, unlabeled


def split_unlabel_data(data, labels, radio):
    """
    将data中数据划分为有标签和无标签的两部分
    :param data:
    :param labels:
    :param radio:
    :return:
    """
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(data, labels, test_size=radio)
    return x_labeled, x_unlabeled, y_labeled, y_unlabeled


def trans_to_d(labels):
    """
    将标签列表转化为{类别：样本集}的格式,
    :param labels:
    :return:
    """
    D = defaultdict(set)
    for index, label in enumerate(labels):
        D[label].add(index)
    return D


def semi_labels_trans(labels: pd.DataFrame, unlabeled: pd.DataFrame):
    d = defaultdict(set)
    for index, row in labels.iterrows():
        d[row.values[0]].add(index)

    d["unlabeled"] = set(unlabeled.index)
    return d


class ReadData:
    """数据读取"""
    def __init__(self, path, norm=True):
        self.path = path
        self.norm = norm
        self.data, self.labels = get_data(self.path, self.norm)

    def get_k_fold(self, k_fold):
        """
        k折交叉验证
        """

        scaled_data, y = self.data.values, self.labels.values

        # K-fold
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(scaled_data)), y)):

            x_train, x_test = scaled_data[train_index], scaled_data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # 训练集划分验证集
            # x_train, x_valid, y_train, y_valid =
            # train_test_split(train_data, train_y, test_size=valid_size, stratify=train_y)
            yield i, x_train, x_test, y_train, y_test


