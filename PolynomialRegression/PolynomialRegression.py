# coding: utf-8
import numpy as np

class PolynomialRegression:
    '''
    只支持numpy数组的输入
    '''
    def __init__(self):
        pass
    def fit(self, x, y):
        '''
        只支持二维数组
        '''
        x= np.hstack((np.ones((len(x), 1)), x))
        x = np.mat(x)
        y = np.mat(y)
        self.w = np.mat((x.T * x).I * x.T * y)
        return self
    def transform(self, x):
        x= np.hstack((np.ones((len(x), 1)), x))
        x = np.mat(x)
        return x * self.w
    def fit_transform(self, x, y):
        x= np.hstack((np.ones((len(x), 1)), x))
        x = np.mat(x)
        y = np.mat(y)
        self.w = (x.T * x).I * x.T * y
        return x * self.w
            


if __name__ == '__main__':
    for i in range(10, 51, 10):
        x = np.random.randint(1, 100, (40, i))
        y = np.random.randint(1, 100, (40, 1))
        reg_result = PolynomialRegression().fit(x, y).transform(x)
        import matplotlib.pyplot as plt
        fig = plt.figure(num = 1, figsize = (15, 8))
        plt.plot(np.arange(len(y)), y, label = 'real_y')
        plt.plot(np.arange(len(reg_result)), reg_result, label = 'pred_y')
        plt.legend()
        plt.show()


