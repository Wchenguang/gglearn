# coding: utf-8

import numpy as np

class NavieBayesClassifier:
    def __init__(self, lamb = 0):
        '''
        func: 初始化
        param: lamb
            type: float
            detail: 0~1之间，用于拉普拉斯平滑
        return: None
        '''
        self.prior_prob_y = {}
        self.prior_prob_x = {}
        self.x_dim = 0
        #拉普拉斯平滑系数
        self.lamb = lamb
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
        self.x_dim = len(x[0])
        y_list = y.tolist()
        y_unique = np.unique(y)
        for val in y_unique:
            self.prior_prob_y[val] = y_list.count(val)/len(y_list)
        y = np.array([y_list])
        xy = np.hstack((x, y.T))
        for d in range(self.x_dim):
            #处理x不同维度
            x_and_y = xy[:, (d,-1)]
            x_unique = np.unique(xy[:, d])
            laplace = len(x_unique)
            self.prior_prob_x[d] = {}
            for yy in y_unique:
                #处理不同的y值
                x_when_yy = x_and_y[x_and_y[:, -1] == yy]
                x_list = x_when_yy[:, 0].tolist()
                self.prior_prob_x[d][yy] = {}
                for xx in x_unique:
                    #获取固定的y下，不同的x的概率
                    self.prior_prob_x[d][yy][xx] = (x_list.count(xx) + self.lamb) / (len(x_list) + laplace * self.lamb)
    def predict(self, x):
        '''
        func: 预测
        param: x
            type: np.ndarray
            detail: 输入x
        return: y
            type: np.ndarray
            detail: 预测y
        '''
        res = {}
        all_pro = 0
        for y_val in self.prior_prob_y:
            res[y_val] = self.prior_prob_y[y_val]
            px_y = 1
            for d in range(self.x_dim):
                print(d, y_val, x[d], self.prior_prob_x[d][y_val][x[d]])
                px_y *= self.prior_prob_x[d][y_val][x[d]]
            res[y_val] *= px_y
            all_pro += res[y_val]
        for y_val in res:
            res[y_val] /= all_pro
