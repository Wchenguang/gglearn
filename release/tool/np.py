def get_real_cols_num(self, arr):
    '''
    func: 获取数组的实际列数
    param: arr
        type: np.ndarray
        detail
    return: col_num
        type: np.int64
        detail: 当数组大小为0，数组为空，列数为0否则不为0
    '''
    if(arr.shape[0] == 0):
        return 0
    else:
        return arr.shape[1]