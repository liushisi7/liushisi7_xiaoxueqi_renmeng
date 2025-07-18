"""
使用最佳参数，训练模型并保存结果
"""

# 导入常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import os
import joblib
from datetime import datetime

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
import warnings 
warnings.filterwarnings('ignore')

# 导入模型和评估工具
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 从自定义模块导入数据处理函数
from data_tezheng import *

def model_pipeline(model):
    '''
    创建整个流程的管道，包括列处理和模型
    '''
    nominal_features = ['城市名称', '包装', '品种', '产地']
    ordinal_features = ['物品尺寸']
    ordinal_features_zd = {'物品尺寸': ['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']}
    
    # 创建预处理器
    preprocessor = pipeline_preprocessor_ColumnTransformer(
        nominal_features=nominal_features, 
        ordinal_features=ordinal_features,
        ordinal_features_zd=ordinal_features_zd
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # 特征预处理
        ('regressor', model)  # 模型
    ])
    return pipeline

def get_model_list():
    '''
    返回要评估的模型列表
    '''
    # 为LightGBM设置特定参数以解决Windows环境问题
    lgbm_params = {
        'n_estimators': 100, 
        'learning_rate': 0.1,
        'max_depth': 10,
        'force_col_wise': True,  # 避免自动选择线程方式
        'verbose': -1,           # 减少输出
        'n_jobs': 1              # 避免多线程问题
    }
    XGB_params = {'learning_rate': 0.1, 'max_depth': 10, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 41}
    
    model_list = [
        LinearRegression(),
        RandomForestRegressor(
            max_depth=None, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            n_estimators=50, 
            n_jobs=1  # 避免Windows多线程问题
        ),
        GradientBoostingRegressor(
            max_depth=None, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            n_estimators=50
        ),
        XGBRegressor(**XGB_params),
        LGBMRegressor(**lgbm_params)
    ]
    return model_list

def save_model(pipeline, model_name, scores):
    '''
    保存模型和评估结果
    '''
    # 创建保存目录
    save_dir = '../models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{save_dir}/{model_name}_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    
    # 保存评估结果
    results = {
        'model': model_name,
        'r2_mean': np.mean(scores['r2']),
        'r2_std': np.std(scores['r2']),
        'rmse_mean': np.mean(scores['rmse']),
        'rmse_std': np.std(scores['rmse']),
        'timestamp': timestamp
    }
    
    # 将结果添加到CSV文件
    results_df = pd.DataFrame([results])
    results_path = f"{save_dir}/model_results.csv"
    
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, index=False)
    
    print(f"模型已保存到: {model_path}")
    print(f"评估结果已保存到: {results_path}")
    
    return model_path

def evaluate_models(X, y):
    '''
    评估所有模型并保存最佳模型
    '''
    models = get_model_list()
    best_score = -float('inf')
    best_model = None
    best_pipeline = None
    best_scores = None
    
    for model in models:
        model_name = model.__class__.__name__
        print(f"\n{'='*50}")
        print(f"模型: {model_name}")
        
        try:
            # 创建并训练pipeline
            pipeline = model_pipeline(model)
            
            # 使用k折交叉验证评估
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # 收集R²分数
            r2_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
            print(f"交叉验证R²分数: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
            
            # 收集RMSE分数
            neg_mse_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-neg_mse_scores)
            print(f"交叉验证RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
            
            # 保存模型评估分数
            scores = {'r2': r2_scores, 'rmse': rmse_scores}
            
            # 如果当前模型是最佳的，则保存
            if np.mean(r2_scores) > best_score:
                best_score = np.mean(r2_scores)
                best_model = model
                best_pipeline = pipeline
                best_scores = scores
            
            # 单独训练完整数据集上的模型并保存
            pipeline.fit(X, y)
            save_model(pipeline, model_name, scores)
            
        except Exception as e:
            print(f"模型 {model_name} 评估失败: {e}")
        print("="*50)
    # 保存最佳模型
    if best_model is not None:
        print(f"\n最佳模型: {best_model.__class__.__name__}, R² = {best_score:.4f}")
        best_path = save_model(best_pipeline, f"BEST_{best_model.__class__.__name__}", best_scores)
        return best_path
    return None

if __name__ == '__main__':
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
