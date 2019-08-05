# coding: utf-8
class LinearRegression:
    def __init__(self, learning_rate = 0.1, method = "normal equation", 
                    derivative_method = "MSE", iteration_times = 100,
                    batch_size = 2):
        '''
        func: 初始化
        param: learning_rate
            type: float
            detail: 学习率
        param: method
            type: str
            detail: 拟合的方法，包括闭式解法（正规方程），梯度下降，随机下降，小批量下降
        param: derivative_method
            type: str
            detail: 指示了在梯度下降系列方法中，所使用的损失函数及求导方法
        param: iteration_times
            type: int
            detail: 迭代次数
        param: batch_size
            type: int
            detail: 小批量下降中的单批次数据量
        return: None
        '''
        self.method = method
        self.derivative_method = derivative_method
        self.learning_rate = learning_rate
        self.iteration_times = iteration_times
        self.batch_size = batch_size
        self.theta = None
        self.theta_history = None
    def _expand_x(self, x):
        '''
        func: 数据集首部扩展一列1，作为权重中偏置值的乘数
        '''
        return np.hstack((x, np.ones(x.shape[0]).reshape(x.shape[0], 1)))
    def _gradient_descent_once(self, theta, x, y, instance_num, learning_rate, 
                               derivative_method = "MSE"):
        '''
        func: 一次梯度下降
        '''
        if(derivative_method == "MSE"):
            gradients = 2/instance_num * x.T.dot(x.dot(theta) - y)
            return theta - learning_rate * gradients
        else:
            return None
    def _fit_normal_equation(self, x, y):
        '''
        func: 正规方程闭式解法
        '''
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    def _fit_gradient_descent(self, x, y, derivative_method, learning_rate,
                             iteration_times):
        '''
        func: 梯度下降法
        '''
        theta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        theta_history = []
        for times in range(iteration_times):
            theta = self._gradient_descent_once(theta, x, y, x.shape[0],
                                                learning_rate, 
                                                derivative_method)
            theta_history.append(theta)
        return theta, theta_history
    def _fit_random_gd(self, x, y, derivative_method, learning_rate, 
                      iteration_times):
        '''
        func: 随机梯度下降法
        '''
        theta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        theta_history = []
        random_index_list = np.arange(x.shape[0])
        for times in range(iteration_times):
            if(0 == len(random_index_list)):
                break
            index = np.random.choice(random_index_list, size = 1, 
                                        replace = False)
            random_index_list = np.setdiff1d(random_index_list, index)
            theta = self._gradient_descent_once(theta, x[index, :], 
                                                y[index, :], 1,
                                                learning_rate, 
                                                derivative_method)
            theta_history.append(theta)
        return theta, theta_history
    def _fit_batch_gd(self, x, y, derivative_method, learning_rate, 
                      iteration_times, batch_size):
        '''
        func: 小批量随机梯度下降法
        '''
        theta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        theta_history = []
        random_index_list = np.arange(x.shape[0])
        for times in range(iteration_times):
            if(0 == len(random_index_list)):
                break
            index_list = np.random.choice(random_index_list, size = batch_size, 
                                         replace = False)
            random_index_list = np.setdiff1d(random_index_list, index_list)
            theta = self._gradient_descent_once(theta, x[index_list, :], 
                                                y[index_list, :], len(index_list), 
                                                learning_rate, 
                                                derivative_method)
            theta_history.append(theta)
        return theta, theta_history
    def fit(self, x, y):
        '''
        func: 拟合
        '''
        x = self._expand_x(x)
        self.theta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        self.theta_history = []
        if(self.method == 'normal equation'):
            self.theta = self._fit_normal_equation(x, y)
        elif(self.method == 'gradient descent'):
            self.theta, self.theta_history = self._fit_gradient_descent(x, y, 
                                            self.derivative_method, 
                                            self.learning_rate,
                                            self.iteration_times)
        elif(self.method == 'random gd'):
            self.theta, self.theta_history = self._fit_random_gd(x, y, 
                                            self.derivative_method, 
                                            self.learning_rate,
                                            self.iteration_times)
        elif(self.method == 'batch gd'):
            self.theta, self.theta_history = self._fit_batch_gd(x, y, 
                                            self.derivative_method, 
                                            self.learning_rate,
                                            self.iteration_times,
                                            self.batch_size)
        return self
    def predict(self, x):
        '''
        func: 预测
        '''
        return self._expand_x(x).dot(self.theta)
    def plot_history(self, x, color = 'r', linewidth = 2):
        '''
        func: 绘制拟合过程中的数据变化
        '''
        expand_x = self._expand_x(x)
        for theta in self.theta_history:
            y = expand_x.dot(theta)
            plt.plot(x, y, color=color, linewidth=linewidth)            
