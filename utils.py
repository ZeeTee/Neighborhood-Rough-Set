# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 21:12
# @Author  : zeetng

from collections import defaultdict
import pandas as pd
from datetime import datetime
import os


def map_to_dps(x, y):
    s = set()
    for i in x:
        for j in y:
            s.add((i, j))
    return s


def group_by_label(label):
    """
    将样本按标签分集合
    :param label: ndarray size: (n, 1)
    :return:
    """

    d = defaultdict(set)
    for i, item in enumerate(label):
        d[int(item[0])].add(i)
    return d


def dateset_format_trans(path):
    df = pd.read_excel(path, header=None)
    df.to_csv(path.split('.')[0]+".csv", header=False, index=False)


def bool_num_attrs(attrs: list, size: int):
    attrs = set(attrs)
    return [True if x in attrs else False for x in range(size)]


def get_log_file(file):
    now_time = datetime.now()
    dir_path = 'logout/{}-{}'.format(now_time.month, now_time.day)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    f = open('{}/{}-{}-{}.txt'.format(dir_path, file, now_time.hour, now_time.minute), 'a')
    return f


if __name__ == '__main__':
    dateset_format_trans("dataset/seeds.xlsx")