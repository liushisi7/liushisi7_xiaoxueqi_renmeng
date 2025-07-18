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
from Data_EDA import data_rename,date_chuli0, data_show_gaikuang,data_show_wenti,auto_eda,data_y_xi_show
from data_tezheng import data_tezheng,data_rename,date_chuli0

def main():
    # 读取数据集
    data = pd.read_csv('../data/US-pumpkins.csv')
    data = date_chuli0(data_rename(data))
    data_show_gaikuang(data)
    data_show_wenti(data)
    for i in data.columns:
        auto_eda(data,i,n_shu=14)
        data_y_xi_show(data,i,'均价')
    # 数据特征分析和特征处理以及选择
    data = data_tezheng(data)
    print(data)






if __name__ == '__main__':
    main()