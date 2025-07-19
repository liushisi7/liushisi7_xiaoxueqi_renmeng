"""
结果可视化图表
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



def plot_actual_vs_predicted(y_test, y_pred, model_name, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='预测点')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想预测线')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} - 实际值 vs 预测值')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{model_name}_actual_vs_predicted.png")
    plt.close()
    print(f"实际值vs预测值图已保存")

def plot_feature_importance(model, preprocessor, model_name, plots_dir, is_tree_based):
    if not is_tree_based:
        return     
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"{model_name} 没有feature_importances_属性")
        return
        
    if len(importances) != len(feature_names):
        print(f"特征重要性长度({len(importances)})与特征名称长度({len(feature_names)})不匹配")
        if len(importances) < len(feature_names):
            feature_names = feature_names[:len(importances)]
        else:
            print("无法修复特征长度不匹配问题，跳过特征重要性图")
            return
        
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(indices))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(top_n), importances[top_indices], color='skyblue')
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha='right')
    plt.xlabel('特征')
    plt.ylabel('重要性得分')
    plt.title(f'{model_name} - 特征重要性 (前{top_n})')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{model_name}_feature_importance.png")
    plt.close()
    print(f"特征重要性图已保存")

# 只绘制部分单棵完整树（从输入到结果）
def plot_tree_visualization(model, model_name, model_type, preprocessor, plots_dir):
        
    # 获取特征名称
    feature_names = preprocessor.get_feature_names_out()
    if not feature_names.any():
        print("无法获取特征名称，使用默认名称")
        if hasattr(model, 'n_features_in_'):
            feature_names = [f'特征_{i}' for i in range(model.n_features_in_)]
        else:
            feature_names = [f'特征_{i}' for i in range(10)]
        
    # 只绘制第一棵完整树（从输入到结果的完整路径）
    print(f"绘制{model_name}的第一棵完整树（从输入到结果）")
    
    if model_type == 'sklearn_ensemble':
        # 提取第一棵树
        if isinstance(model, RandomForestRegressor) and hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            tree_to_plot = model.estimators_[0]  # 取第一棵树
        elif isinstance(model, GradientBoostingRegressor) and hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            tree_to_plot = model.estimators_[0, 0]  # 取第一棵树
        else:
            print(f"{model_name} 无可用决策树用于可视化")
            return
        
        # 计算合适的图形大小（根据树深度动态调整）
        max_depth = tree_to_plot.get_depth()
        fig_width = 15 + (max_depth - 3) * 5  # 深度每增加1，宽度增加5
        fig_height = 10 + (max_depth - 3) * 3  # 深度每增加1，高度增加3
        figsize = (min(fig_width, 50), min(fig_height, 30))  # 限制最大尺寸
        
        plt.figure(figsize=figsize)
        plot_tree(
            tree_to_plot,
            max_depth=None,  # 不限制深度，完整显示整棵树
            filled=True,
            feature_names=feature_names,
            rounded=True,
            precision=2,
            fontsize=6 if max_depth > 5 else 8  # 深度大时减小字体
        )
        plt.title(f'{model_name} - 第一棵完整树（从输入到结果）')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_single_tree_complete.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"{model_name} 完整树图已保存")
    
    else:
        print(f"暂时绘制{model_name}的树结构可视化未实现")

# SHAP值和部分依赖图函数保持不变
def plot_shap_values(model, X_sample, preprocessor, model_name, plots_dir, is_tree_based):
    if not is_tree_based:
        print(f"{model_name} 不是树模型，跳过SHAP值计算")
        return
        
    try:
        X_processed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()
        
        if X_processed.shape[1] != len(feature_names):
            print(f"处理后的数据维度({X_processed.shape[1]})与特征名称数量({len(feature_names)})不匹配")
            if X_processed.shape[1] < len(feature_names):
                feature_names = feature_names[:X_processed.shape[1]]
            else:
                print("无法修复维度不匹配问题，跳过SHAP图")
                return
        
        explainer = shap.TreeExplainer(model, feature_names=feature_names)
        shap_values = explainer.shap_values(X_processed)
        
        if shap_values is None:
            print(f"SHAP值计算失败，跳过SHAP图")
            return
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'{model_name} - SHAP特征重要性')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_shap_summary_bar.png", dpi=300)
        plt.close()
        print(f"SHAP特征重要性条形图已保存")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
        plt.title(f'{model_name} - SHAP值分布')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_shap_summary_bee.png", dpi=300)
        plt.close()
        print(f"SHAP蜂群图已保存")
        
        for i in range(min(2, len(X_sample))):
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[i,:], 
                X_processed[i,:],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/{model_name}_shap_force_sample_{i}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP force plot (样本 {i}) 已保存")
            
    except Exception as e:
        print(f"绘制SHAP图时出错: {str(e)}")
