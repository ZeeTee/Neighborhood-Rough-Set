# -*- coding: utf-8 -*-
# @Time    : 2020/4/13 22:12
# @Author  : zeetng

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'dataset/wine.csv'

# 读取数据
raw_data = pd.read_csv(DATA_PATH, header=None)

# 分离标签
labels = raw_data.pop(0)
data = pd.DataFrame(raw_data.values)

# 标准化
sca = MinMaxScaler()
sca_data = sca.fit_transform(data)
scaled_data = pd.DataFrame(sca_data)

scaled_data["label"] = labels.values
sns.set(style="ticks")
sns.pairplot(scaled_data, hue="label")
plt.show()


def get_data():
    return labels.values, scaled_data


def get_spilt_data(ratio):
    """

    :param ratio: 未标记数据比例
    :return:
    """

    # 按比例划分标记与未标记数据
    unlabeled = scaled_data.sample(frac=ratio)
    label = labels.drop(list(unlabeled.index))
    labeled = scaled_data.drop(unlabeled.index, axis=0)

    return labeled, label.values, unlabeled


# if __name__ == '__main__':
#     get_spilt_data(0.1)


