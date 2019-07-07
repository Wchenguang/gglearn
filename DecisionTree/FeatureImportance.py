# coding: utf-8

import numpy as np
import math
'''
熵的计算
'''
def entropy(y_values):
    e = 0
    unique_vals = np.unique(y_values)
    for val in unique_vals:
        p = np.sum(y_values == val)/len(y_values)
        e += (p * math.log(p, 2))
    return -1 * e

'''
条件熵的计算
'''
def entropy_condition(x_values, y_values):
    ey = entropy(y_values)
    ey_condition = 0
    xy = np.hstack((x_values, y_values))
    unique_x = np.unique(x_values)
    for x_val in unique_x:
        px = np.sum(x_values == x_val) / len(x_values)
        xy_condition_x = xy[np.where(xy[:, 0] == x_val)]
        ey_condition_x = entropy(xy_condition_x[:, 1])
        ey_condition += (px * ey_condition_x)
    return ey - ey_condition

'''
信息增益比：摒弃了选择取值多的特征为重要特征的缺点
'''
def entropy_condition_ratio(x_values, y_values):
    return entropy_condition(x_values, y_values) / entropy(x_values)


'''
基尼指数计算
'''
def gini(y_values):
    g = 0
    unique_vals = np.unique(y_values)
    for val in unique_vals:
        p = np.sum(y_values == val)/len(y_values)
        g += (p * p)
    return 1 - g

'''
按照x取值的基尼指数的计算
'''
def gini_condition(x_values, y_values):
    g_condition = {}
    xy = np.hstack((x_values, y_values))
    unique_x = np.unique(x_values)
    for x_val in unique_x:
        xy_condition_x = xy[np.where(xy[:, 0] == x_val)]
        xy_condition_notx = xy[np.where(xy[:, 0] != x_val)]
        g_condition[x_val] = len(xy_condition_x)/len(x_values) * gini(xy_condition_x[:, 1]) + len(xy_condition_notx)/len(x_values) * gini(xy_condition_notx[:, 1])
    return g_condition


