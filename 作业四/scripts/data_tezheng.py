#导入常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
#设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 无视警告
import warnings 
# warnings.filterwarnings('always') # 显示所有警告
warnings.filterwarnings('ignore') # 忽略所有警告

from Data_EDA import data_rename,date_chuli0
def data_tezheng(data):
    """
    数据特征分析和特征处理以及选择
    data: 数据集
    return: 特征处理后的,仅仅包含所需要的特征x和目标y的数据集
    *注意在时间维度上，去除日期会导致相似/相同 记录的重复，可能会造成数据泄露---需要对y进行处理，去除重复的y（删除/均值）
    """
    data = data_Hot_Deck_Imputation(data) # 热卡填补法
    # 重新包装中N与E的数量比过大，E只有5条记录，且存在至少两个特征的缺失值，故不考虑重新包装特征
    data.drop(columns=['重新包装'], inplace=True)
    # 删除全部为空的列
    data = data.dropna(axis=1, how='all')
    # 删除空值数量过多的列---该类数据的信息量太少，对后续分析无意义
    data = data.dropna(thresh=len(data)*0.2, axis=1) #thresh参数:指定保留的行或列中至少应包含的非缺失值的数量(可以容忍的缺失值数量)
    data['均价']= (data['最低价格'] + data['最高价格']) / 2
    # 删除 颜色
    data.drop(columns=['颜色','主要最低价','主要最高价','最低价格','最高价格'], inplace=True)
    # 去除 物品尺寸 为空值的记录
    data = data.dropna(axis=0, how='any')
    # 去除异常值
    q1 = data['均价'].quantile(0.25)
    q3 = data['均价'].quantile(0.75)
    iqr = q3 - q1
    data = data[~((data['均价'] < (q1 - 1.5 * iqr)) | (data['均价'] > (q3 + 1.5 * iqr)))]
    # data = data[[ '月份', '星期', '物品尺寸', '包装', '品种', '产地', '均价']]
    x=data[[ '月份', '星期', '物品尺寸', '包装', '品种', '产地', '均价']]
    y=data['均价']
    # print(data.info())
    return x, y

def data_Hot_Deck_Imputation(data):
    """
    热卡填补法-手动查找
    """
    print('检查填充前空值情况：',data['产地'].isnull().sum()) 
    # 使用空值记录中其他存在的特征相同的记录的品种 填充，如 城市名称、包装、产地、（颜色、尺寸）------ 使用其中出现频率最高的品种属性填充
    data['产地'].fillna(data.groupby(['城市名称','包装','物品尺寸','品种'])['产地'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
    print('检查填充后是否还有空值：',data['产地'].isnull().sum())
    print('检查填充前空值情况：',data['品种'].isnull().sum()) 
    # 使用空值记录中其他存在的特征相同的记录的品种 填充，如 城市名称、包装、产地、（颜色、尺寸）------ 使用其中出现频率最高的品种属性填充
    data['品种'].fillna(data.groupby(['城市名称','包装','物品尺寸','产地','最低价格'])['品种'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
    print('检查填充后是否还有空值：',data['品种'].isnull().sum()) # 检查填充后是否还有空值
    data['品种'].fillna(data.groupby(['城市名称','包装','物品尺寸','产地'])['品种'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
    print('放宽填要求后是否还有空值：',data['品种'].isnull().sum()) # 品牌成功填充
    return data


if __name__ == '__main__':
    # 读取数据集
    data = pd.read_csv('../data/US-pumpkins.csv')
    data=date_chuli0(data_rename(data))
    # 数据特征分析和特征处理以及选择
    data = data_tezheng(data)
    print(data)
    # 保存特征处理后的数据集
