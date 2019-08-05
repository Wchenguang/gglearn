# coding: utf-8

import math
import operator
from collections import Counter
import numpy as np

def entropy(y_values):
    '''
    func: 计算熵
    param: y_values
        type: np.ndarray
        detail: y的值
    return: e
        type: np.float64
        detail: 计算熵，反映了y的混乱程度
    '''
    e = 0
    unique_vals = np.unique(y_values)
    for val in unique_vals:
        p = np.sum(y_values == val)/len(y_values)
        e += (p * math.log(p, 2))
    return -1 * e


def entropy_condition(x_values, y_values):
    '''
    func: 计算信息增益
    param: x_values
        type: np.ndarray
        detail: x的值
    param: y_values
        type: np.ndarray
        detail: y的值
    return: e
        type: np.float64
        detail: 计算信息熵与条件熵，做差可得信息增益，反映了得知x后对y混乱程度的改善
    '''
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

def entropy_condition_ratio(x_values, y_values):
    '''
    func: 计算信息增益比
    param: x_values
        type: np.ndarray
        detail: x的值
    param: y_values
        type: np.ndarray
        detail: y的值
    return: e
        type: np.float64
        detail: 计算信息熵与条件熵，做差可得信息增益，反映了得知x后对y混乱程度的改善
    '''
    return entropy_condition(x_values, y_values) / entropy(x_values)

def gini(y_values):
    '''
    func: 计算基尼指数
    param: y_values
        type: np.ndarray
        detail: y的值
    return: g
        type: np.float64
        detail: 计算基尼指数
    '''
    g = 0
    unique_vals = np.unique(y_values)
    for val in unique_vals:
        p = np.sum(y_values == val)/len(y_values)
        g += (p * p)
    return 1 - g

def gini_condition(x_values, y_values):
    '''
    func: 计算条件基尼指数
    param: y_values
        type: np.ndarray
        detail: y的值
    return: g
        type: np.float64
        detail: 计算条件基尼指数
    '''
    g_condition = {}
    xy = np.hstack((x_values, y_values))
    unique_x = np.unique(x_values)
    for x_val in unique_x:
        xy_condition_x = xy[np.where(xy[:, 0] == x_val)]
        xy_condition_notx = xy[np.where(xy[:, 0] != x_val)]
        g_condition[x_val] = len(xy_condition_x)/len(x_values) * gini(xy_condition_x[:, 1]) + len(xy_condition_notx)/len(x_values) * gini(xy_condition_notx[:, 1])
    return g_condition


