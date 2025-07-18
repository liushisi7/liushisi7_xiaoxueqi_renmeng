"""
使用最佳参数（beat_model_cv.py得到），使用k折交叉验证，训练模型，并保存模型

"""

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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression  # 替换为回归模型
from sklearn.metrics import mean_squared_error, r2_score  # 使用回归评估指标

# from data_tezheng import pipeline_preprocessor_ColumnTransformer,data_tezheng   #pipeline_preprocessor_ColumnTransformer,data_tezheng等
from data_tezheng import *

def model_pipeline(model):
    '''
    整个流程的管道，包括列处理，模型
    '''
    nominal_features = ['城市名称', '包装', '品种', '产地']
    ordinal_features = ['物品尺寸']
    ordinal_features_zd = {'物品尺寸': ['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']}
    
    # 创建预处理器
    preprocessor = pipeline_preprocessor_ColumnTransformer(
        # numerical_features=numerical_features, 
        nominal_features=nominal_features, 
        ordinal_features=ordinal_features,
        ordinal_features_zd=ordinal_features_zd
    )
    pipeline = Pipeline([
    ('preprocessor', preprocessor),  # 特征预处理
    ('regressor', model)  # 模型
    ])
    return pipeline
 
if __name__ == '__main__':
    data = pd.read_csv('../data/US-pumpkins.csv')
    data = date_chuli0(data_rename(data))
    # 数据特征分析和特征处理以及选择
    data = data_tezheng(data)
    data = pd.DataFrame(data)
    y = data['均价']
    X = data.drop(columns=['均价'])    
    # model = sk.linear_model.LinearRegression(copy_X= True, fit_intercept= True, n_jobs= -1)
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    model=RandomForestRegressor(max_depth= None, min_samples_leaf= 1, min_samples_split=2, n_estimators= 50)
    pipeline = model_pipeline(model)
    pipeline.fit(X, y)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # 使用R²评分作为回归评估指标
    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    
    print("\n" + "="*50)
    print(f"交叉验证R²分数: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    print("="*50)
    
    # 均方误差
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)  # 将负MSE转换为RMSE
    print(f"交叉验证RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    print("="*50)
