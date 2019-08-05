# coding: utf-8
def euler_dist(self, x1, x):
    '''
    func: 计算2个向量组每一行的欧拉距离
    '''
    return np.sqrt(np.multiply(x-x1, x-x1).sum())
    
def manhattan_distance(self, x1, x):
    '''
    func: 计算2个向量组每一行的曼哈顿距离
    '''
    return np.abs(x-x1).sum()
  
def time_gapper(seconds, basetime = 0):
    '''
    func: 将秒数时间戳转化为每一天的不同时段
    param: seconds
        type: int
        detail: 距离基准时间所过的秒数
    param: basetime
        type: int
        detail: 基准秒数
    return: time_interval
        type: np.ndarray
        detail: 转化为时段，支持批量转换
    '''
    seconds_a_day = 24 * 60 * 60
    seconds_an_hour = 60 * 60
    gaptime_a_day = (seconds - basetime) % seconds_a_day
    hours = gaptime_a_day // seconds_an_hour + gaptime_a_day % seconds_an_hour / seconds_an_hour
    res = np.full(seconds.shape, "time_a_day")
    if((hours >= 23).any()):
        res[hours >= 23] = 'mid night'
    if((hours < 7).any()):
        res[hours < 7] = 'mid night'
    if((hours>=7) & (hours < 11)).any():
        res[(hours>=7) & (hours < 11)] = 'morning'
    if((hours>=11) & (hours < 13)).any():
        res[(hours>=11) & (hours < 13)] = 'noon'
    if((hours>=13) & (hours < 18)).any():
        res[(hours>=13) & (hours < 18)] = 'afternoon'
    if((hours>=18) & (hours < 23)).any():
        res[(hours>=18) & (hours < 23)] = 'night'
    return res.reshape(-1, 1)
    
def age_gapper(frame, age_col):
    '''
    func: 将年龄转化为年龄段
    param: frame
        type: pandas.DataFrame
        detail
    param: age_col
        type: str
        detail: 年龄所在列名
    return: age_interval
        type: np.ndarray
        detail: 支持批量转换
    '''
    res = np.full(frame[age_col].shape, 'wealthy')
    res[frame[age_col] < 26] = "student"
    res[(frame[age_col] >= 26) & (frame[age_col] < 41)] = "adult"
    res[frame[age_col] >= 41] = "old"
    return res
    
