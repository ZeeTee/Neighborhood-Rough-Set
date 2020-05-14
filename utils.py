# -*- coding: utf-8 -*-
# @Time    : 2020/3/27 21:12
# @Author  : zeetng

from collections import defaultdict


def map_to_dps(x, y):
    s = set()
    for i in x:
        for j in y:
            s.add((i, j))
    return s


def group_by_label(label):
    # 将样本按标签分集合

    d = defaultdict(set)
    for i, item in enumerate(label):
        d[int(item[0])].add(i)
    return d
