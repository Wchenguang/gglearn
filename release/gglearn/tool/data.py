# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def pareto(dataframe, x_col, y_col, figsize=(10, 6)):
    '''
    func: DataFrame中的多个列，返回其帕托雷数据（二八原理）
    param: dataframe
        type: pandas.DataFrame
        detail
    param: x_col
        type: list
        detail: 自变量x的索引列表
    param: y_col
        type: int/...
        detail: 因变量索引
    param: figsize
        type: tuple
        detail: 每行图的大小
    return: None
    '''
    y = dataframe[y_col].copy()
    y.index = dataframe[x_col]
    y = y.sort_values(ascending = False)
    print("origin")
    print(y)
    cum = 1.0 * y.cumsum() / y.sum()
    print("cum")
    print(cum)
    return cum[cum[:] < 0.9].index

def outlier_cleaner(dataframe, method="box"):
    '''
    func: 异常值清理
    param: dataframe
        type: pandas.DataFrame
        detail
    param: method
        type: str
        detail: 清理方法包括箱线图，3sigma
    return: dataframe
        type: pandas.DataFrame
        detail
    '''
    columns = dataframe.columns
    des = dataframe.describe()
    cleaned_frame = dataframe.copy()
    if(method == "box"):
        for col in columns:
            box_max = des[col]['75%'] + 1.5 * (des[col]['75%'] - des[col]['25%'])
            box_min = des[col]['25%'] - 1.5 * (des[col]['75%'] - des[col]['25%'])
            cleaned_frame = cleaned_frame.loc[(cleaned_frame[col] < box_max) & 
                                         (cleaned_frame[col] > box_min)]
        return cleaned_frame
    elif(method == "3sigma"):
        for col in columns:
            sigma_max = des[col]['mean'] + 3 * des[col]['std']
            sigma_min = des[col]['mean'] - 3 * des[col]['std']
            box_min = des[col]['25%'] - 1.5 * (des[col]['75%'] - des[col]['25%'])
            cleaned_frame = cleaned_frame.loc[(cleaned_frame[col] < sigma_max) & 
                                         (cleaned_frame[col] > sigma_min)]
        return cleaned_frame
    
class Interpolation:
    '''
    插值方法
    '''
    def __init__(self, method = "newton"):
        '''
        func: 初始化
        param: method
            type: str
            detail: 包含牛顿插值方法
        return: None
        '''
        self.method = method
        self.x = None
        self.y = None
        self.weight_matrix = None
        self.weight = None
    def fit(self, x, y):
        '''
        func: 拟合
        '''
        self.x = x
        self.y = y
        self.weight_matrix = np.zeros((len(x), len(x)))
        self.weight_matrix[:, 0] = y
        for col in range(1, len(x)):
            for row in range(col, len(x)):
                self.weight_matrix[row][col] = (self.weight_matrix[row - 1][col - 1] - 
                                         self.weight_matrix[row][col - 1])/ (x[row - col]
                                                                  - x[row])
        self.weight = self.weight_matrix.diagonal().copy()
        self.weight[0] = 1
    def transform(self, x):
        '''
        func: 插值处理
        '''
        cutted_x = self.x[:-1]
        temp = np.array([x for i in range(len(self.x) - 1)])
        temp -= cutted_x.reshape(-1, 1)
        temp = temp.cumprod(axis = 0)
        temp = np.vstack((np.full((1, len(x)), self.y[0]), temp))
        return np.dot(self.weight, temp)
    def newton_add(self, x, y):
        '''
        func: 牛顿插值特有的，可以增量添加新的数据点
        '''
        if(self.method == "newton"):
            pass
           
def normalization(data, method = "mean std"):
    '''
    func: 数据归一化
    param: method
        type: str
        detail: 数据归一化方法，包含最大最小，均值标准差，以及定标方法
    '''
    if(method == "mean std"):
        return (data - data.mean()) / (data.std())
    elif(method == "max min"):
        return (data - data.min()) / (data.max() - data.min())
    else:
        #decimal scaling
        return data / (10 ** np.ceil(np.log10(np.abs(data).max())))
 
def pca_contribution(data):
    '''
    func: 获取pca数据降维后，各个特征的贡献度，根据贡献度加和比例，选取尽量少的特征进行保留
    param: data
        type: DataFrame/ndarray
        detail
    return: ratio
        type: list
        detail: 返回百分数形式的，各个特征的贡献度，并从大到小排序
    '''
    pca = PCA(copy = True)
    pca.fit(data)
    return pca.explained_variance_ratio_
    
def stratified_sample(data, columns, train_size, verify_size):
    '''
    func: 从DataFrame形式存储的大样本中，抽取小样本的训练集与验证集，采用分层抽样
    param: data
        type: DataFrame
        detail
    param: columns
        type: list
        detail: 分层抽样所需考虑的属性列
    param: train_size, verify_size
        type: float
        detail: 0~1之间，表明抽取数据的比例
    return: train, verify
        type: DataFrame
        detail
    '''
    train = verify = temp = None
    split = StratifiedShuffleSplit(n_splits = 1, test_size = train_size)
    for trainIndex, testIndex in split.split(data, data[columns]):
        train = data.loc[testIndex]
        temp = data.loc[trainIndex]
    split = StratifiedShuffleSplit(n_splits = 1, test_size = verify_size / (1 - train_size))
    for trainIndex, testIndex in split.split(temp, temp[columns]):
        verify = data.loc[testIndex]
    return train, verify