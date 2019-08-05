# coding: utf-8
def boxplot_multi_cols(dataframe, per_line = 5, figsize = (8,3), columns = None):
    '''
    func: DataFrame中的多个列，展示其箱线图
    param: dataframe
        type: pandas.DataFrame
        detail
    param: per_line
        type: int
        detail: 每行展示的图个数，每行的图在一个fig中
    param: figsize
        type: tuple
        detail: 每行图的大小
    param: columns
        type: list
        detail: 为None则绘制所有列，否则绘制特定列
    return: None
    '''
    if(columns == None):
        columns = dataframe.columns
    if(len(columns) % per_line == 0):
        line_num = len(columns) // per_line
    else:
        line_num = len(columns) // per_line  + 1
    for line in range(line_num):
        part_cols = columns[line*per_line : (line+1)*per_line]
        ax = plt.figure(figsize = figsize).gca()
        part_df = dataframe[part_cols]
        part_df.plot.box(ax = ax, grid = True)
        
def hist_multi_cols(dataframe, figsize = (5, 3)):
    '''
    func: DataFrame中的多个列，展示其频率分布图
    param: dataframe
        type: pandas.DataFrame
        detail
    param: figsize
        type: tuple
        detail: 每行图的大小
    return: None
    '''
    columns = dataframe.columns
    for col in columns:
        fig = plt.figure(figsize=figsize)
        plt.hist(dataframe[col], label = col)
        plt.legend()
        plt.show()

def plot_simple_data_convert(x, y):
    '''
    func: 绘制数据函数转换后的数据分布，包括平方，根号，对数，取差4种方法
    '''
    plt.figure(figsize=(10, 20))
    ##
    y1 = np.multiply(y, y)
    plt.subplot(4, 2, 1)
    plt.scatter(x, y1)
    plt.title('x^2 scatter')
    plt.subplot(4, 2, 2)
    plt.hist(y1)
    plt.title('x^2 hist')
    ##
    y2 = np.sqrt(y)
    plt.subplot(4, 2, 3)
    plt.scatter(x, y2)
    plt.title('x^0.5 scatter')
    plt.subplot(4, 2, 4)
    plt.hist(y2)
    plt.title('x^0.5 hist')
    ##
    y3 = np.log(y)
    plt.subplot(4, 2, 5)
    plt.scatter(x, y3)
    plt.title('lnx scatter')
    plt.subplot(4, 2, 6)
    plt.hist(y3)
    plt.title('lnx hist')
    ##
    y4 = np.diff(y)
    plt.subplot(4, 2, 7)
    plt.scatter(x[1:], y4)
    plt.title('x_k+1 - x_k scatter')
    plt.subplot(4, 2, 8)
    plt.hist(y4)
    plt.title('x_k+1 - x_k hist')
    plt.show()
    
def plot_nv1_scatter(data, xcols, ycol, figsize = (6,4)):
    '''
    func: 绘制多个x同一y的散点图
    param: data
        type: DataFrame
        detail
    param: xcols
        type: list
        detail: x属性列
    param: ycol
        type: int/...
        detail: y属性列
    param: figsize
        type: tuple
        detail: 图大小
    return: None
    '''
    for xcol in xcols:
        plt.figure(figsize=figsize)
        plt.scatter(x = data[xcol], y = data[ycol])
        plt.title(xcol + " vs " + ycol)
        plt.ylabel(ycol)
        plt.xlabel(xcol)
        plt.show()