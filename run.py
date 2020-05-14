# -*- coding: utf-8 -*-
# @Time    : 2020/3/20 22:15
# @Author  : zeetng

from dis_neighborhood_rough_set import DP_NRS
import pandas as pd
import logging
import preprocessing
import utils


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)


# data = pd.read_csv('/Volumes/Seagate Backup Plus Drive/UCI/wdbc/wdbc.csv',
#                   header=None)
# prepare data
# label = data[34].copy()
# data = data.drop(34, axis=1)
# data = pd.DataFrame(
#     {0: [0.0909, 0, 0.4091, 0.6364, 1.0000, 0.9091, 0.9545, 0.6818],
#      1: [1.0000, 0.3750, 0, 0.7500, 0.3750, 0.5000, 0.6250, 0.6250],
#      2: [1, 1, 1, 1, 2, 2, 2, 1],
#      3: ['Setosa', 'Setosa', 'Virginica', 'Virginica', 'Virginica', 'Versicolor', 'Versicolor', 'Versicolor']}
#     )
# label = data[3].copy()
# data = data.drop([0, 1], axis=1)
# data = pd.DataFrame(data.values, columns=[x for x in range(30)])

# sca = MinMaxScaler()
# sca_data = sca.fit_transform(data)
# sca_data = pd.DataFrame(sca_data)

# enc = OrdinalEncoder()
# label = enc.fit_transform(label.values.reshape(-1, 1))
labeled_data, labels, unlabeled_data = preprocessing.get_spilt_data(0.1)
# n_sample, n_attrs = data.shape

red = set()  # 约间集
A = list(labeled_data.columns)  # 属性集A

d_nrs = DP_NRS(labeled_data, labels, unlabeled_data, 0.13)
d_nrs.run_reduction(100)
# d_nrs.run_backward()

