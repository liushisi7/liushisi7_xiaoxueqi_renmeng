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
from model import *

def main():
    # 载入数据并进行预处理
    data = pd.read_csv('../data/US-pumpkins.csv')
    data = date_chuli0(data_rename(data))
    data = data_tezheng(data)
    data = pd.DataFrame(data)
    
    # 分离特征和目标变量
    y = data['均价']
    X = data.drop(columns=['均价'])
    
    # 评估所有模型并保存结果
    best_model_path = evaluate_models(X, y)
    
    if best_model_path:
        print(f"\n最佳模型已保存到: {best_model_path}")
        
        # 使用最佳模型进行预测示例
        best_model = joblib.load(best_model_path)
        # 此处可以添加使用模型进行预测的代码






if __name__ == '__main__':
    main()