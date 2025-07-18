'''
通过网格搜索对于模型的参数进行调优，找到各个模型最优参数，
输出各个模型 针对不同目标参数 搜索得到的最优参数，并将交叉验证结果保存到JSON文件
'''

# 导入常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import json
# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 无视警告
import warnings
warnings.filterwarnings('ignore') # 忽略所有警告

from data_tezheng import data_tezheng, date_chuli0, data_rename
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm
import xgboost
from model import model_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from data_tezheng import *

# 读取数据集
data = pd.read_csv('../data/US-pumpkins.csv')
data = date_chuli0(data_rename(data))
data = data_tezheng(data)

y = data['均价']
X = data.drop(columns=['均价'])

# 定义参数网格
param_grid = {
    'LinearRegression': 
    {'regressor__fit_intercept': [True, False], 'regressor__n_jobs': [None, -1]},
    "DecisionTreeRegressor":
    {'regressor__criterion': ['squared_error', 'absolute_error'], 'regressor__splitter': ['best', 'random'], 'regressor__max_depth': [None, 10, 20], 'regressor__min_samples_split': [2, 5, 10], 'regressor__min_samples_leaf': [1, 2, 4]},
    'RandomForestRegressor': 
    {'regressor__n_estimators': [10, 50, 100], 'regressor__max_depth': [None, 10, 20], 'regressor__min_samples_split': [2, 5, 10], 'regressor__min_samples_leaf': [1, 2, 4]},
    'LightGBM':
    {'regressor__n_estimators': [10, 50, 100], 'regressor__learning_rate': [0.01, 0.1, 1.0], 'regressor__max_depth': [5, 10, 20], 'regressor__min_child_samples': [1, 5, 20]},
    'XGBoost':
    {'regressor__n_estimators': [10, 50, 100], 'regressor__learning_rate': [0.01, 0.1, 1.0], 'regressor__max_depth': [3, 6, 9], 'regressor__min_child_weight': [1, 3, 5]}
}

# 创建模型字典
models = {
    'LinearRegression': LinearRegression,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'LightGBM': lightgbm.LGBMRegressor,
    'XGBoost': xgboost.XGBRegressor
}



best_params = {}

# 网格搜索寻找最佳参数
for model_name, model_class in models.items():
    if model_name in param_grid:
        model = model_pipeline(model_class())#管道中包含了预处理和模型
        grid_search = GridSearchCV(model, param_grid[model_name], cv=3, scoring='r2')
        grid_search.fit(X, y)
        best_params[model_name] = grid_search.best_params_
        print(f"{model_name} 最佳参数: {grid_search.best_params_}")
        print(f"{model_name} 最佳得分: {grid_search.best_score_:.4f}")

# 保存最佳参数为 JSON 文件
with open('best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4, ensure_ascii=False)
