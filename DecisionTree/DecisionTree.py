# In[125]:
import numpy as np
from FeatureImportance import *

import operator
class DTNode:
    '''
    决策树节点
    '''
    def __init__(self, x, y, default_label, split_val = None, cart_cols = []):
        self.children = []
        if(len(y) != 0):
            self.label = Counter(y.reshape(1, -1).tolist()[0]).most_common(1)[0][0]
        else:
            self.label = default_label
        self.next_split_index = None
        self.split_val = split_val
        self.x = x.copy()
        self.y = y.copy()
        self.xy = np.hstack([x, y])
        self.default_label = default_label
        self.cart_cols = cart_cols
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_xy(self):
        return self.xy
    def get_children(self):
        return self.children
    def get_label(self):
        return self.label
    def get_next_split_index(self):
        return self.next_split_index
    def get_split_val(self):
        return self.split_val
    def _get_real_cols_num(self, arr):
        if(arr.shape[0] == 0):
            return 0
        else:
            return arr.shape[1]
    def _get_x_and_xval(self, calculate_method, threshold):
        '''
        根据所选方法及阈值，计算信息增益（比）,选择目标特征, 并计算目标特征取值种类
        '''
        res = {}
        for col_index in range(self._get_real_cols_num(self.x)):
            res[col_index] = calculate_method(self.x[:, col_index].reshape(-1, 1), self.y.reshape(-1, 1))
        if(operator.eq(res, {})):
            return None, None
        else:
            target = sorted(res, key=res.__getitem__, reverse=True)[0]
            if(res[target] < threshold):
                return None, None
            else:
                return target, np.unique(self.x[:, target])
    def _cart_get_x_and_feature(self, calculate_method, threshold, cols):
        '''
        根据所选方法及阈值，计算基尼指数以及均方差，返回最佳划分特征，以及最佳划分特征值
        '''
        res = {}
        for col_index in cols:
            res[col_index] = calculate_method(self.x[:, col_index].reshape(-1, 1), self.y.reshape(-1, 1))
            target = sorted(res[col_index], key=res[col_index].__getitem__)[0]
            res[col_index] = (target, res[col_index][target])
        if(operator.eq(res, {})):
            return None, None
        else:
            target = sorted(res, key = lambda k: res[k][1])[0]
            if(res[target][1] < threshold):
                return None, None
            else:
                return target, res[target][0]
        
    def _build_exit(self):
        if(len(np.unique(self.y)) == 1):
            self.label = np.unique(self.y)[0]
            return True
        elif(operator.eq(self.x.tolist(), [])):
            self.label = self.default_label
            return True
        else:
            return False
        
    def _cart_build_exit(self):
        if(operator.eq(self.x.tolist(), [])):
            self.label = self.default_label
            return True
        else:
            return False
        
    def build_children(self, method, threshold):
        '''
        检测退出条件
        '''
        if(self._build_exit()):
            return 
        '''
        构建子节点
        '''
        if(method == 'information gain'):
            x_index, x_val = self._get_x_and_xval(entropy_condition, threshold)
        else:
            #method == 'information gain ratio'
            x_index, x_val = self._get_x_and_xval(entropy_condition_ratio, threshold)
        '''
        无需分割
        label置为当前最多的label值
        ？
        '''
        if(x_index == None):
            #self.label = self.default_label
            return
        self.next_split_index = x_index
        for val in x_val:
            splited_xy = self.xy[self.xy[:, x_index] == val]
            splited_xy = np.delete(splited_xy, [x_index], axis = 1)
            self.children.append(DTNode(splited_xy[:, :-1], splited_xy[:, -1].reshape(-1, 1), self.default_label, val))
            
    def cart_build_children(self, method, threshold):
        '''
        检测退出条件
        '''
        if(self._cart_build_exit()):
            return 
        '''
        构建子节点
        '''
        if(method == 'gini'):
            x_index, x_val = self._cart_get_x_and_feature(gini_condition, threshold, self.cart_cols)
        if(x_index == None):
            return
        self.next_split_index = x_index
        splited_left_xy = self.xy[self.xy[:, x_index] == x_val]
        splited_right_xy = self.xy[self.xy[:, x_index] != x_val]
        next_cart_cols = self.cart_cols.copy()
        next_cart_cols.remove(x_index)
        self.children.append(DTNode(splited_left_xy[:, :-1], splited_left_xy[:, -1].reshape(-1, 1), 
                                    self.default_label, x_val, cart_cols = next_cart_cols))
        self.children.append(DTNode(splited_right_xy[:, :-1], splited_right_xy[:, -1].reshape(-1, 1), 
                                    self.default_label, x_val, cart_cols = next_cart_cols))


# In[128]:


from collections import Counter
class DecisionTree:
    '''
    决策树
    '''
    def __init__(self, method, threshold):
        self.x = None
        self.y = None
        self.root = None
        self.threshold = threshold
        self.default_label = None
        self.method = method
        if(method == 'ID3'):
            self.feature_selection_method = "information gain"
        elif(method == 'cart clf'):
            self.feature_selection_method = "gini"
        else:
            #method == 'C4.5'
            self.feature_selection_method = "information gain ratio"
    def fit(self, x, y):
        self.x = x
        self.y = y
        '''
        筛选默认label，即训练集中频率最高的label
        '''
        self.default_label = Counter(self.y.reshape(1, -1).tolist()[0]).most_common(1)[0][0]
        '''
        宽度遍历建立决策树
        '''
        self.root = DTNode(x, y, self.default_label, cart_cols = list(range(self.x.shape[1])))
        queue = [self.root]
        while(len(queue) > 0):
            node = queue.pop(0)
            if('information' in self.feature_selection_method):
                node.build_children(self.feature_selection_method, self.threshold)
            else:
                node.cart_build_children(self.feature_selection_method, self.threshold)
            queue += node.get_children()
    def show(self):
        '''
        展示各个节点的信息
        '''
        queue = [self.root]
        while(len(queue) > 0):
            node = queue.pop(0)
            print('==============')
            print('node label:', node.get_label())
            print('node split_val', node.get_split_val())
            print('node next_split_index:', node.get_next_split_index())
            print('xy:')
            print(node.get_xy())
            queue += node.get_children()
        


# In[129]:


xy = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2], [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0], [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0], 
             [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0], [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]]).T
dt = DecisionTree(method = 'cart clf', threshold = 0.01)
dt.fit(xy[:, :-1], xy[:, -1].reshape(-1, 1))
dt.show()


# In[130]:


xy = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2], [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0], [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0], 
             [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0], [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]]).T
dt = DecisionTree(method = 'ID3', threshold = 0.1)
dt.fit(xy[:, :-1], xy[:, -1].reshape(-1, 1))
dt.show()

