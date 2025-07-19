"""
增强保存机制：确保所有可视化图表完整保存 - 只绘制单棵完整树
"""

# 导入常用库（保持不变）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
import shap
from sklearn.tree import plot_tree
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import lightgbm as lgb
import warnings

# 设置显示中文（保持不变）
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 导入模型和评估工具（保持不变）
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 从自定义模块导入数据处理函数（保持不变）
from data_tezheng import pipeline_preprocessor_ColumnTransformer, date_chuli0, data_rename, data_tezheng
from show_jieguo_fx import *

# 创建保存图表和结果的目录（保持不变）
def create_model_directories(model_name, timestamp):
    base_dir = f'../models/{model_name}_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    plots_dir = f'{base_dir}/plots'
    os.makedirs(plots_dir, exist_ok=True)
    return base_dir, plots_dir

# model_pipeline、get_model_list、calculate_metrics等函数保持不变

def model_pipeline(model):
    nominal_features = ['城市名称', '包装', '品种', '产地']
    ordinal_features = ['物品尺寸']
    ordinal_features_zd = {'物品尺寸': ['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']}
    
    preprocessor = pipeline_preprocessor_ColumnTransformer(
        nominal_features=nominal_features, 
        ordinal_features=ordinal_features,
        ordinal_features_zd=ordinal_features_zd
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipeline

def get_model_list():
    return [{'name': 'LinearRegression','class': LinearRegression,'params': {},'is_tree_based': False,'model_type': 'linear'},
        {'name': 'RandomForestRegressor','class': RandomForestRegressor,'params': {
                'max_depth': 5,  # 限制深度确保单棵树可完整显示
                'min_samples_leaf': 5,  # 增加叶节点样本数简化树结构
                'min_samples_split': 10,  # 增加分裂样本数简化树结构
                'n_estimators': 50, 'n_jobs': 1,'random_state': 42},'is_tree_based': True,'model_type': 'sklearn_ensemble'
        },
        {'name': 'GradientBoostingRegressor','class': GradientBoostingRegressor,
        'params': {'max_depth': 5, 'min_samples_leaf': 5,'min_samples_split': 10,'n_estimators': 50,'random_state': 42},
        'is_tree_based': True,'model_type': 'sklearn_ensemble'},
        {'name': 'XGBRegressor','class': XGBRegressor,
         'params': {'learning_rate': 0.1,'max_depth': 5,'min_child_weight': 5,'n_estimators': 100, 'random_state': 42},
        'is_tree_based': True,'model_type': 'xgboost'},
        {'name': 'LGBMRegressor','class': LGBMRegressor,
        'params': {'n_estimators': 100, 'learning_rate': 0.1,'max_depth': 5,'min_child_samples': 5, 'force_col_wise': True,'verbose': -1,'n_jobs': 1,'random_state': 42 },
        'is_tree_based': True,'model_type': 'lightgbm'}]

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    return {'r2': r2,'rmse': rmse,'mae': mae,'mape': mape}

def save_prediction_results(X_test, y_test, y_pred, model_name, timestamp, base_dir):
    results_df = X_test.copy()
    results_df['实际值'] = y_test
    results_df['预测值'] = y_pred
    results_df['差值'] = y_pred - y_test
    results_df['绝对差值'] = np.abs(y_pred - y_test)
    results_df['相对误差(%)'] = np.abs((y_pred - y_test) / y_test.replace(0, np.nan)) * 100
    
    results_path = f"{base_dir}/{model_name}_{timestamp}_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存到: {results_path}")
    return results_df

# 更新保存函数，调用优化后的树可视化函数
def save_model_and_visualizations(pipeline, model_name, model_type, is_tree_based, 
                                 scores, fold_scores, X_test, y_test, y_pred, X_sample):
    timestamp = datetime.now().strftime("%Y%m%d")
    base_dir, plots_dir = create_model_directories(model_name, timestamp)
    
    # 保存模型
    model_path = f"{base_dir}/{model_name}_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)
    
    # 保存预测结果
    results_df = save_prediction_results(X_test, y_test, y_pred, model_name, timestamp, base_dir)
    
    # 绘制核心可视化图表
    plot_actual_vs_predicted(y_test, y_pred, model_name, plots_dir)
    
    # 获取预处理器和模型
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['regressor']
    
    # 绘制特征重要性图
    plot_feature_importance(model, preprocessor, model_name, plots_dir, is_tree_based)
    
    # 如果是树模型，绘制单棵完整树、SHAP值和部分依赖图
    if is_tree_based:
        # 调用优化后的树可视化函数（只绘制单棵完整树）
        plot_tree_visualization(model, model_name, model_type, preprocessor, plots_dir)
        
        # 绘制SHAP值图
        plot_shap_values(model, X_sample, preprocessor, model_name, plots_dir, is_tree_based)
        
    
    # 保存汇总评估结果（保持不变）
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
    
    summary_path = "../models/model_summary_results.csv"
    results_df_summary = pd.DataFrame([results])
    if os.path.exists(summary_path):
        existing_results = pd.read_csv(summary_path)
        updated_results = pd.concat([existing_results, results_df_summary], ignore_index=True)
        updated_results.to_csv(summary_path, index=False)
    else:
        results_df_summary.to_csv(summary_path, index=False)
    
    if fold_scores is not None:
        fold_results_path = f"{base_dir}/{model_name}_{timestamp}_fold_results.csv"
        pd.DataFrame(fold_scores).to_csv(fold_results_path, index=False)
        print(f"详细折评估结果已保存到: {fold_results_path}")
    
    print(f"模型已保存到: {model_path}")
    print(f"汇总评估结果已保存到: {summary_path}")
    print(f"可视化图表已保存到: {plots_dir}")
    
    return model_path, base_dir

