#!/usr/bin/env python
# coding: utf-8

# # AdaBoost梯度提升算法
# 
# ## 算法步骤与原理
# 
# 1. 训练 $ m $ 个弱学习分类器，分类器有相同的接口
# $$ 
# G_{m}(x) : \mathcal{X} \rightarrow\{x_{1},x_{2} \dots\}
#  $$
# 2. 假设数据有均匀的权值分布，即每个样本在分类器中作用相同，$ n $个实例的权重为
# $$ 
# D_{1}=\left(w_{11}, \cdots, w_{1 i}, \cdots, w_{1 N}\right), \quad w_{1 i}=\frac{1}{N}, \quad i=1,2, \cdots, N
#  $$
#  对于$ m $个分类器而言，有$ m \times n $个权重
# 3. 进入迭代循环，在每一次循环中进行如下操作
# 
#     3.1 计算$ m $个分类器在加权数据集上的分类错误率
#     $$ 
#     e_{m}=P\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{C_{n}\left(x_{i}\right) \neq y_{i}} w_{m i}
#     $$
#     3.2 计算每个分类器的权重$ alpha_{m} $，该权重表明，每个单独分类器在最终分类器中的重要程度
#     $$ 
#     \alpha_{m}=\frac{1}{2} \log \frac{1-e_{m}}{e_{m}}
#     $$
#     * 由上式可知，随着分类器的误差率的减小，其权重值越大
#     
#    3.3 更新数据集的权重分布
#     
#     $$ 
# \begin{array}{c}{D_{m+1}=\left(w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}\right)} \\ {w_{m+1, i}=\frac{w_{m i}}{Z_{m}} \exp \left(-\alpha_{m} (y_{i}== G_{m}\left(x_{i})\right)\right), \quad i=1,2, \cdots, N}\end{array}
#  $$
#  $$ 
# Z_{m}=\sum_{i=1}^{N} w_{m i} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)
#  $$
#  
#     * 由上式可知
#          $$ 
# w_{m+1, i}=\left\{\begin{array}{ll}{\frac{w_{m i}}{Z_{m}} \mathrm{e}^{-\alpha_{m}},} & {G_{m}\left(x_{i}\right)=y_{i}} \\ {\frac{w_{m i}}{Z_{m}} \mathrm{e}^{\alpha_{m}},} & {G_{m}\left(x_{i}\right) \neq y_{i}}\end{array}\right.
#  $$
#      预测错误的实例，权重提升。预测正确的实例，权重下降。
#     
#  
# 

# In[111]:

'''
import numpy as np

class testClf:
    def __init__(self, thresold):
        self.thresold = thresold
        self.x = None
        self.y = None
    def fit(self, x, y):
        self.x = x
        self.y = y
        return self
    def predict(self, x):
        y = x.copy()
        less_index = np.where(y[:, 0] < self.thresold)
        greater_index = np.where(y[:, 0] > self.thresold)
        y[less_index] = 1
        y[greater_index] = -1
        return y
    def fit_predict(self, x, y):
        return self.fit(x, y).predict(x)
'''
        
'''
test_x = np.arange(10).reshape(-1, 1)
test_y = np.array([1,1,1,-1,-1,-1,1,1,1,-1]).reshape(-1, 1)
tc = testClf(2.5)
print(tc.fit_predict(test_x, test_y))
'''


# In[130]:


import numpy as np
import matplotlib.pyplot as plt

class AdaBoost:
    def __init__(self, clf_list, iteration_times):
        '''
        分类器需要有相同的fit，predict接口用于训练及预测
        '''
        self.clf_list = clf_list
        self.iteration_times = iteration_times
        self.x_weight_matrix = None
        self.clf_weight = None
    def _em(self, y_predict, y, x_weight):
        y_predict_flag = (y_predict != y).astype(int)
        return np.multiply(y_predict_flag, x_weight).sum()
    def _am(self, em):
        return np.log((1- em) / em) * 0.5
    def _update_x_weight(self, y_predict, y, am, x_weight):
        y_predict_flag = (y_predict == y).astype(int)
        y_predict_flag[np.where(y_predict_flag[:, 0] == 0)] = -1
        zm_array = np.multiply(np.exp(y_predict_flag * am * -1),
                                    x_weight)
        zm_array = zm_array / zm_array.sum()
        return zm_array
    def _fit_once(self, x, y, x_weight, clf_weight):
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
        m = len(self.clf_list)
        n = x.shape[0]
        if(0 == n or 0 == m):
            return
        self.x_weight = np.full((n, 1), 1/n)
        self.clf_weight = np.full((m, 1), 1/m)
        for i in range(self.iteration_times):
            self._fit_once(x, y, self.x_weight, self.clf_weight)
    def transform(self, x):
        if(self.clf_list == None or 0 == len(self.clf_list)):
            return None
        res = self.clf_weight[0] * self.clf_list[0].predict(x)
        for index in range(1, len(self.clf_list)):
            res += (self.clf_weight[index] * 
                            self.clf_list[index].predict(x))
        return res


# In[131]:

'''
test_x = np.arange(10).reshape(-1, 1)
test_y = np.array([1,1,1,-1,-1,-1,1,1,1,-1]).reshape(-1, 1)
adaboost = AdaBoost([testClf(2.5), testClf(8.5), testClf(5.5), ], 1)
adaboost.fit(test_x, test_y)
predict = adaboost.transform(test_x)
predict[np.where(predict[:, 0] < 0)] = -1
predict[np.where(predict[:, 0] >= 0)] = 1
print('predict')
print(predict)
print('truth')
print(test_y)
'''

# * 与书中P140-P41结果相符
# * 书中分类器G3计算应为错误

# In[ ]:




