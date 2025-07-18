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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
    返回要评估的模型配置列表
    '''
    model_configs = [
        {
            'name': 'LinearRegression',# 线性回归
            'class': LinearRegression,
            'params': {}
        },
        {
            'name': 'RandomForestRegressor',# 随机森林 
            'class': RandomForestRegressor,
            'params': {
                'max_depth': None, 
                'min_samples_leaf': 1, 
                'min_samples_split': 2, 
                'n_estimators': 50, 
                'n_jobs': 1  # 避免Windows多线程问题
            }
        },
        {
            'name': 'GradientBoostingRegressor', # 梯度提升
            'class': GradientBoostingRegressor,
            'params': {
                'max_depth': None, 
                'min_samples_leaf': 1, 
                'min_samples_split': 2, 
                'n_estimators': 50
            }
        },
        {
            'name': 'XGBRegressor',
            'class': XGBRegressor,
            'params': {
                'learning_rate': 0.1, 
                'max_depth': 10, 
                'min_child_samples': 20, 
                'n_estimators': 200, 
                'num_leaves': 41
            }
        },
        {
            'name': 'LGBMRegressor',
            'class': LGBMRegressor,
            'params': {
                'n_estimators': 100, 
                'learning_rate': 0.1,
                'max_depth': 10,
                'force_col_wise': True,  # 避免自动选择线程方式
                'verbose': -1,           # 减少输出
                'n_jobs': 1              # 避免多线程问题
            }
        }
    ]
    return model_configs

def calculate_metrics(y_true, y_pred):
    """计算并返回多种评估指标"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算MAPE，处理y_true为0的情况
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def save_model(pipeline, model_name, scores, fold_scores=None):
    '''
    保存模型和评估结果
    '''
    # 创建保存目录
    save_dir = '../models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d")
    model_path = f"{save_dir}/{model_name}_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    
    # 保存汇总评估结果
    results = {
        'model': model_name,
        'r2_mean': np.mean(scores['r2']),
        'r2_std': np.std(scores['r2']),
        'rmse_mean': np.mean(scores['rmse']),
        'rmse_std': np.std(scores['rmse']),
        'mae_mean': np.mean(scores['mae']),
        'mae_std': np.std(scores['mae']),
        'mape_mean': np.mean(scores['mape']),
        'mape_std': np.std(scores['mape']),
        'timestamp': timestamp
    }
    
    # 将汇总结果添加到CSV文件
    results_df = pd.DataFrame([results])
    results_path = f"{save_dir}/model_summary_results.csv"
    
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, index=False)
    
    # 保存每折详细结果
    if fold_scores is not None:
        fold_results_df = pd.DataFrame(fold_scores)
        fold_results_path = f"{save_dir}/{model_name}_{timestamp}_fold_results.csv"
        fold_results_df.to_csv(fold_results_path, index=False)
        print(f"详细折评估结果已保存到: {fold_results_path}")
    
    print(f"模型已保存到: {model_path}")
    print(f"汇总评估结果已保存到: {results_path}")
    
    return model_path

def run_all_list_models(X, y):
    '''
    评估所有模型并保存最佳模型
    '''
    model_configs = get_model_list()
    best_score = -float('inf')
    best_model = None
    best_pipeline = None
    best_scores = None
    
    for config in model_configs:
        model_name = config['name']
        print(f"\n{'='*50}")
        print(f"模型: {model_name}")
        # 初始化模型
        model = config['class'](**config['params'])
            
        # 创建并训练pipeline
        pipeline = model_pipeline(model)
            
        # 使用k折交叉验证评估
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
        # 存储每一折的评估结果
        fold_scores = []
        all_metrics = {'r2': [],'rmse': [],'mae': [],'mape': []}
        # 执行手动交叉验证以获取详细的预测结果
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            pipeline.fit(X_train, y_train)
            
            # 预测并评估
            y_pred = pipeline.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            
            # 存储当前折的评估结果
            fold_metrics = {'fold': fold + 1}
            fold_metrics.update(metrics)
            fold_scores.append(fold_metrics)
            
            # 收集所有折的指标
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            print(f"折 {fold + 1}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        # 打印汇总结果
        print(f"交叉验证R²: {np.mean(all_metrics['r2']):.4f} (±{np.std(all_metrics['r2']):.4f})")
        print(f"交叉验证RMSE: {np.mean(all_metrics['rmse']):.4f} (±{np.std(all_metrics['rmse']):.4f})")
        print(f"交叉验证MAE: {np.mean(all_metrics['mae']):.4f} (±{np.std(all_metrics['mae']):.4f})")
        print(f"交叉验证MAPE: {np.mean(all_metrics['mape']):.2f}% (±{np.std(all_metrics['mape']):.2f}%)")
        
        # 如果当前模型是最佳的，则保存
        if np.mean(all_metrics['r2']) > best_score:
            best_score = np.mean(all_metrics['r2'])
            best_model = model
            best_pipeline = pipeline
            best_scores = all_metrics
        
        # 单独训练完整数据集上的模型并保存
        pipeline.fit(X, y)
        save_model(pipeline, model_name, all_metrics, fold_scores)
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
    best_model_path = run_all_list_models(X, y)
    
    if best_model_path:
        print(f"\n最佳模型已保存到: {best_model_path}")

