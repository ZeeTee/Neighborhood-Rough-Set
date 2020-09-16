# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 16:30
# @Author  : zeetng
# realize rough set based semi-supervised feature selection via ensemble selector
from sklearn.semi_supervised import LabelPropagation
import numpy as np
from preprocessing import ReadData, split_unlabel_data


PATH = ""
K = 10

reader = ReadData(PATH)
for r in np.arange(0.1, 1, 0.1):
    for i, x_train, x_test, y_train, y_test in reader.get_k_fold(K):
        # 分割有无标签的数据集
        x_labeled, x_unlabeled, y_labeled, y_unlabeled = split_unlabel_data(x_train, y_train, r)

        # LPA生成标签
        lp_model = LabelPropagation()
        lp_model.fit(x_labeled, y_labeled)
        y_inductive = lp_model.predict(x_unlabeled)

        # n is number of decision classes
        n = y_labeled.max()+1

        A = set()


