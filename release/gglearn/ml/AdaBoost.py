# coding: utf-8

import numpy as np

class AdaBoost:
    '''
    detail: AdaBoost算法将多个弱分类器组合成为强分类器
    '''
    def __init__(self, clf_list, iteration_times):
        '''
        func: 初始化
        param: clf_list 
            type: list 
            detail: 算法内部使用的弱分类器列表，内部使用的分类器需要
            有相同的fit，predict接口用于训练及预测
        param: iteration_times
            type: int
            detail: 算法更新的迭代次数
        return: None
        '''
        self.clf_list = clf_list
        self.iteration_times = iteration_times
        self.x_weight_matrix = None
        self.clf_weight = None
    def _em(self, y_predict, y, x_weight):
        '''
        func: 计算分类误差率
        param: y_predict
            type: numpy.ndarray
            detail: 算法对当前x的预测值
        param: y
            type: numpy.ndarray
            detail: y的实际值
        param: x_weight
            type:numpy.ndarray
            detail: 当前数据集x的权重值
        return: em
            type: np.float64
            detail: 返回该分类器对x的分类误差率
        '''
        y_predict_flag = (y_predict != y).astype(int)
        return np.multiply(y_predict_flag, x_weight).sum()
    def _am(self, em):
        '''
        func: 计算分类器权重
        param: em
            type: np.float64
            detail: 该分类器的分类误差率
        return: am
            type: np.float64
            detail: 通过分类误差率，计算该分类器的权重即系数，降低出错更多分类器权重，反之增加
        '''
        return np.log((1- em) / em) * 0.5
    def _update_x_weight(self, y_predict, y, am, x_weight):
        '''
        func: 更新数据集x权重
        param: y_predict
            type: np.ndarray
            detail: 算法对当前x的预测值
        param: y
            type: np.ndarray
            detail: y的实际值
        param: am
            type: np.float64
            detail: 当前弱分类器的系数
        param: x_weight
            type: np.ndarray
            detail: 当前数据集x的权重值
        return: zm_array
            type: np.ndarray
            detail: 每次选择一个弱分类器，根据该分类器的计算数据，更新数据集x的权重值
        '''
        y_predict_flag = (y_predict == y).astype(int)
        y_predict_flag[np.where(y_predict_flag[:, 0] == 0)] = -1
        zm_array = np.multiply(np.exp(y_predict_flag * am * -1),
                                    x_weight)
        zm_array = zm_array / zm_array.sum()
        return zm_array
    def _fit_once(self, x, y, x_weight, clf_weight):
        '''
        func: 拟合数据一次
        param: x
            type: np.ndarray
            detail: 输入数据集x
        param: y
            type: np.ndarray
            detail: y的实际值
        param: x_weight
            type: np.ndarray
            detail: 当前数据集x的权重值
        param: clf_weight
            type: np.ndarray
            detail: 当前弱分类器的权重值
        return: None
            detail: 迭代一次，每次遍历所有弱分类器
        '''
        for index in range(len(self.clf_list)):
            clf = self.clf_list[index]
            y_predict = clf.fit_predict(x, y)
            em = self._em(y_predict, y, x_weight)
            am = self._am(em)
            x_weight = self._update_x_weight(y_predict, y, am, x_weight)
            clf_weight[index] = am
            '''print('em', em, 'am', am)
            print('更新后权重')
            print(x_weight)'''
    def fit(self, x, y):
        '''
        func: 拟合数据
        param: x
            type: np.ndarray
            detail: 输入数据集x
        param: y
            type: np.ndarray
            detail: y的实际值
        return: None
            detail: 拟合数据
        '''
        m = len(self.clf_list)
        n = x.shape[0]
        if(0 == n or 0 == m):
            return
        self.x_weight = np.full((n, 1), 1/n)
        self.clf_weight = np.full((m, 1), 1/m)
        for i in range(self.iteration_times):
            self._fit_once(x, y, self.x_weight, self.clf_weight)
    def transform(self, x):
        '''
        func: 预测
        param: x
            type: np.ndarray
            detail: 输入数据集x
        return: res
            type: np.ndarray
            detail: 算法对x的预测值y
        '''
        if(self.clf_list == None or 0 == len(self.clf_list)):
            return None
        res = self.clf_weight[0] * self.clf_list[0].predict(x)
        for index in range(1, len(self.clf_list)):
            res += (self.clf_weight[index] * 
                            self.clf_list[index].predict(x))
        return res
