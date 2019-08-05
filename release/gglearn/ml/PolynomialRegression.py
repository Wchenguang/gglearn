# coding: utf-8

import numpy as np

class PolynomialRegression:
    '''
    func: 多项书回归，实现为线性回归
    '''
    def __init__(self):
        pass
    def fit(self, x, y):
        '''
        func: 拟合
        param: x
            type: np.ndarray
            detail: 输入x
        param: y
            type: np.ndarray
            detail: 输入y
        return: None
        '''
        x= np.hstack((np.ones((len(x), 1)), x))
        x = np.mat(x)
        y = np.mat(y)
        self.w = np.mat((x.T * x).I * x.T * y)
        return self
    def transform(self, x):
        '''
        func: 预测
        param: x
            type: np.ndarray
            detail: 输入x
        return: y
            type: np.ndarray
            detail: 预测y
        '''
        x= np.hstack((np.ones((len(x), 1)), x))
        x = np.mat(x)
        return x * self.w
    def fit_transform(self, x, y):
        '''
        func: 拟合且预测
        param: x
            type: np.ndarray
            detail: 输入x
        param: y
            type: np.ndarray
            detail: 输入y
        return: y
            type: np.ndarray
            detail: 预测y
        '''
        x= np.hstack((np.ones((len(x), 1)), x))
        x = np.mat(x)
        y = np.mat(y)
        self.w = (x.T * x).I * x.T * y
        return x * self.w
            