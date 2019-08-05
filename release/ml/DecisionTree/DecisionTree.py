from FeatureImportance import *

class DTNode:
    '''
    detail: 决策树节点
    '''
    def __init__(self, x, y, default_label, split_val = None, cart_cols = []):
        '''
        func: 初始化
        param: x
            type: np.ndarray
            detail: 输入数据x
        param: y
            type: np.ndarray
            detail: 输入数据y
        param: default_label
            type: object
            detail: 当该节点该节点的默认类别，当该节点无法向下划分子节点，则为默认类别
        param: split_val
            type: np.float64
            detail: 该节点被划分时所依据的值
        param: cart_cols
            type: list
            detail: x值的索引列表，在cart生成时有效，反映了在这一节点中可用于gini选择的列
        return: None
        '''
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
        '''
        func: 返回实际列数
        param: arr
            type: np.ndarray
            detail: 任意数组
        return: col
            type: np.int64
            detail: 当数据大小为0，则实际列数为0，表明没有数据
        '''
        if(arr.shape[0] == 0):
            return 0
        else:
            return arr.shape[1]
    def _get_x_and_xval(self, calculate_method, threshold):
        '''
        func: 用于熵生成的特征选择
        param: calculate_method
            type: function
            detail: 熵相关的计算方法包含熵，熵增益，熵增益比
        param: threshold
            type: float
            detail: 熵阈值，当所有计算结果比它小时，该节点应为叶节点不再向下划分
        return: target, index | None, None
            type: int, np.ndarray
            detail: 返回划分属性列在x中的索引，以及该列的取值范围
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
        func: 用于cart生成的特征选择
        param: calculate_method
            type: function
            detail: 基尼相关的计算方法包含基尼指数，条件基尼指数
        param: threshold
            type: float
            detail: 基尼阈值，当所有计算结果比它小时，该节点应为叶节点不再向下划分
        param: cols
            type: list
            detail: 当前节点可以用于属性选择的列，与熵方法不同在于，每次选择一属性后，子节点不可再选择该属性
                    而熵方法保证不会选到祖先节点的属性，因为哪些属性皆为熵最小的状态
        return: target, feature | None, None
            type: int, list
            detail: 与熵方法不同的是，该方法还需返回最佳划分特征值
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
        '''
        func: 熵方法建树的退出检测条件
        return: None
            detail: 熵方法建树的退出检测条件
        '''
        if(len(np.unique(self.y)) == 1):
            self.label = np.unique(self.y)[0]
            return True
        elif(operator.eq(self.x.tolist(), [])):
            self.label = self.default_label
            return True
        else:
            return False
        
    def _cart_build_exit(self):
        '''
        func: cart方法建树的退出检测条件
        return: None
            detail: cart方法建树的退出检测条件
        '''
        if(operator.eq(self.x.tolist(), [])):
            self.label = self.default_label
            return True
        else:
            return False
        
    def build_children(self, method, threshold):
        '''
        func: 建立子树
        param: method
            type: str
            detail: 熵相关的计算方法包含熵，熵增益，熵增益比
        param: threshold
            type: float
            detail: 熵阈值，当所有计算结果比它小时，该节点应为叶节点不再向下划分
        return: None
        '''
        if(self._build_exit()):
            '''
            检测退出条件
            '''
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
        默认label置为当前最多的label值
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
        func: 建立子树
        param: method
            type: str
            detail: 基尼相关的计算方法包含基尼指数，条件基尼指数
        param: threshold
            type: float
            detail: 基尼阈值，当所有计算结果比它小时，该节点应为叶节点不再向下划分
        return: None
        '''
        if(self._cart_build_exit()):
         
        '''
        检测退出条件
        '''
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

class DecisionTree:
    '''
    func: 决策树
    '''
    def __init__(self, method, threshold):
        '''
        func: 初始化
        param: method
            type: str
            detail: 建树方法
        param: threshold
            type: float
            detail: 建树节点选择时的阈值
        return: None
            detail: 不同method选择不同计算方法
        '''
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
        func: 展示各个节点的信息
        return: None
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
