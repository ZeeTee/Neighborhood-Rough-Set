# -*- coding: utf-8 -*-
# @Time    : 2020/4/13 22:12
# @Author  : zeetng

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ---------------包含数据预处理的各种操作-------------------------

# scaled_data["label"] = labels.values
# sns.set(style="ticks")
# sns.pairplot(scaled_data, hue="label")
# plt.show()

# 数据集路径
DATA_PATH = 'dataset/wine.csv'


def get_data(path=DATA_PATH, norm=True):
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


def get_partial_labeled_data(ratio, path=DATA_PATH, norm=True, seed=34):
    """
    获取部分标记数据
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
    labeled = raw.drop(unlabeled.index, axis=0)

    return labeled, label.values, unlabeled


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


