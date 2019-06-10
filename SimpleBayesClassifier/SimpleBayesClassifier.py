
# coding: utf-8


import numpy as np

class SimpleBayesClassifier:
    def __init__(self, lamb = 0):
        self.prior_prob_y = {}
        self.prior_prob_x = {}
        self.x_dim = 0
        #拉普拉斯平滑系数
        self.lamb = lamb
    def fit(self, x, y):
        '''
        x是二维ndarray数组
        y是一维ndarray数组
        x，y长度相同
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
        x是一维数组
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


# In[43]:


#利用书中的实例测试
if __name__ == '__main__':
    xy = [[1,4,-1],
        [1,5,-1],
        [1,5,1],
        [1,4,1],
        [1,4,-1],
        [2,4,-1],
        [2,5,-1],
        [2,5,1],
        [2,6,1],
        [2,6,1],
        [3,6,1],
        [3,5,1],
        [3,5,1],
        [3,6,1],
        [3,6,-1]]
    xy = np.array(xy)


    sb_clf = SimpleBayesClassifier(1)
    sb_clf.fit(xy[:, (0,1)], xy[:, -1])

    print('x prob', sb_clf.prior_prob_x)
    print('y prob', sb_clf.prior_prob_y)





