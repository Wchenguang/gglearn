# coding: utf-8

import numpy as np

class KmeansClassifier:
    def __init__(self, k, distance_method = "o", random_select = True, 
                    plot = False):
        '''
        func: 初始化
        param: k
            type: int
            detail: 聚类个数
        param: distance_method
            type: str
            detail: 距离计算方法，包含欧拉距离，曼哈顿距离
        param: random_select
            type: bool
            detail: 是否随机选择质心
        param: plot
            type: bool
            detail: 是否展示拟合过程中的状态，只适用二维数据
        return: None
        detail: 设定质心得功能未定
        '''
        self.k = k
        self.distance_method = distance_method
        self.random_select = random_select
        self.plot = plot
        if(distance_method == "o"):
            self._dist = self._euler_dist
        else:
            self._dist = self._manhattan_distance
            
    def _euler_dist(self, x1, x):
        '''
        func: 计算欧拉距离
        param: x1, x
            type: np.ndarray
            detail: 无顺序要求
        return: distance
            type: np.float64
            detail:欧拉距离
        '''
        return np.sqrt(np.multiply(x-x1, x-x1).sum())
    
    def _manhattan_distance(self, x1, x):
        '''
        func: 计算曼哈顿距离
        param: x1, x
            type: np.ndarray
            detail: 无顺序要求
        return: distance
            type: np.float64
            detail: 曼哈顿距离
        '''
        return np.abs(x-x1).sum()
    
    def _get_nearest(self, x, center_list, dist):
        '''
        func: 获取一个点x的最近质心
        param: x
            type: np.ndarray
            detail: 一个向量，表示点x在各个维度的坐标
        param: center_list
            type: list
            detail: 质心列表
        param: dist
            type: function
            detail: 距离计算方法
        return: index
            type: int
            detail: 返回最近的质心在center_list中的索引
        '''
        dists = []
        for center in center_list:
            dists.append(dist(x, center))
        return dists.index(min(dists))
    
    def _fit(self, x, y, dist, x_center_index_list, center_list):
        '''
        func: 拟合
        param: x
            type: np.ndarray
            detail: 输入数据x
        param: y
            type: np.ndarray
            detail: 输入数据y
        param: dist
            type: function
            detail: 距离计算方法
        param: x_center_index_list
            type: np.ndarray
            detail: 数据x中，每一个数据点距离最近的质心，其索引构成该列表
        param: center_list
            type: np.ndarray
            detail: 质心列表，若指定则不需要随机生成初始质心
        return: flag, x_center_index_list, center_list
            type: bool, np.ndarray, np.ndarray
            detail: flag用于指示数据是否发生变化，不变则停止拟合
        '''
        xy_map = np.hstack((x, x_center_index_list, x_center_index_list))
        for row in xy_map:
            row[-1] = self._get_nearest(row[:-2], center_list, dist)
        flag = np.all(xy_map[:, -1] == xy_map[:, -2])
        return flag, xy_map[:, -1].reshape(-1, 1), center_list
    
    def _random_center_list(self, x, k):
        '''
        func: 初始生成随机质心
        param: x
            type: np.ndarray
            detail: 输入数据x
        param: k
            type: int
            detail: 质心个数
        return: center_list
            type: list
            detail
        '''
        center_list = np.zeros((x.shape[1], k))
        for col in range(x.shape[1]):
            col_max = np.max(x[:, col])
            col_min = np.min(x[:, col])
            center_list[col, :] = col_min + (col_max - col_min) * np.random.rand(1, k)
        return center_list.T
    
    def _updata_center_list(self, x, x_center_index_list, center_list):
        '''
        func: 更新质心列表
        param: x
            type: np.ndarray
            detail: 输入数据x
        param: x_center_index_list
            type: np.ndarray
            detail: 数据x中，每一个数据点距离最近的质心，其索引构成该列表
        param: center_list
            type: np.ndarray
            detail: 质心列表
        return: center_list
            type: ndarray
            detail: 质心列表
        '''
        new_center_list = []
        for index in range(len(center_list)):
            part_x = x[np.where(
                    x_center_index_list[:, -1] == index)]
            if(0 != part_x.size):
                new_center_list.append(np.mean(part_x, axis = 0))
            else:
                new_center_list.append(np.zeros(part_x.shape[1]))
        return new_center_list
    
    def _plot(self, x, x_center_index_list, center_list):
        '''
        func: 绘制数据分布
        param: x
            type: np.ndarray
            detail: 输入数据x
        param: x_center_index_list
            type: np.ndarray
            detail: 数据x中，每一个数据点距离最近的质心，其索引构成该列表
        param: center_list
            type: np.ndarray
            detail: 质心列表
        return: None
            detail
        '''
        center_array = np.array(center_list)
        for index in range(len(center_list)):
            part_x = x[np.where(
                    x_center_index_list[:, -1] == index)]
            plt.scatter(part_x[:, 0], part_x[:, 1])
        plt.scatter(center_array[:, 0], center_array[:, 1], marker = "+")
        plt.show()
    
    def fit(self, x, y, center_list = None):
        '''
        func: 拟合
        param: x
            type: np.ndarray
            detail: 输入数据x
        param: y
            type: np.ndarray
            detail: 输入数据y
        param: center_list
            type: np.ndarray
            detail: 质心列表，若指定则不需要随机生成初始质心
        return: None
            detail
        '''
        if not center_list:
            center_list = self._random_center_list(x, self.k)
        x_center_index_list = np.zeros(x.shape[0]).reshape(-1, 1)
        flag = False
        while(True):
            flag, x_center_index_list, center_list = self._fit(x, 
                                        y, self._dist, x_center_index_list, center_list)
            if(flag):
                break
            center_list = self._updata_center_list(x, x_center_index_list,
                                                      center_list)
            if(self.plot):
                self._plot(x, x_center_index_list, center_list)
        return self
