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
def load_latest_model(model_type):
    """加载指定类型的最新模型"""
    model_dir = '../models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') and model_type in f]
    
    if not model_files:
        raise FileNotFoundError(f"找不到{model_type}模型文件")
    
    # 按时间戳排序获取最新模型
    model_files.sort(reverse=True)
    latest_model_path = os.path.join(model_dir, model_files[0])
    return joblib.load(latest_model_path)

def get_feature_names(pipeline):
    """从预处理管道中获取特征名称"""
    preprocessor = pipeline.named_steps['preprocessor']
    
    # 获取数值特征名称
    numeric_features = preprocessor.transformers_[0][2]
    
    # 获取分类特征名称
    categorical_features = preprocessor.transformers_[1][2]
    categorical_processor = preprocessor.transformers_[1][1]
    if hasattr(categorical_processor, 'get_feature_names_out'):
        categorical_features = categorical_processor.get_feature_names_out(categorical_features)
    
    # 获取有序特征名称
    ordinal_features = preprocessor.transformers_[2][2]
    ordinal_processor = preprocessor.transformers_[2][1]
    if hasattr(ordinal_processor, 'get_feature_names_out'):
        ordinal_features = ordinal_processor.get_feature_names_out(ordinal_features)
    
    return list(numeric_features) + list(categorical_features) + list(ordinal_features)

def visualize_and_save_tree(model, feature_names, model_type, tree_index=0):
    """可视化树结构并保存为图片和文本"""
    # 创建保存目录
    tree_dir = f'../tree_visualizations/{model_type}'
    os.makedirs(tree_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 15))
    
    if model_type == 'RandomForestRegressor':
        # 随机森林树可视化
        plot_tree(model.estimators_[tree_index], 
                  feature_names=feature_names, 
                  filled=True, rounded=True,
                  fontsize=10, max_depth=3)
        plt.title(f'随机森林 - 树 {tree_index}')
        
        # 保存文本表示
        tree_text = export_text(model.estimators_[tree_index], 
                               feature_names=feature_names,
                               max_depth=10)
        with open(f"{tree_dir}/{model_type}_tree_{tree_index}.txt", "w") as f:
            f.write(tree_text)
    
    elif model_type == 'XGBRegressor':
        # XGBoost树可视化
        ax = plt.gca()
        xgb.plot_tree(model, num_trees=tree_index, ax=ax, rankdir='LR')
        plt.title(f'XGBoost - 树 {tree_index}')
        
        # 保存文本表示
        tree_text = model.get_booster().get_dump()[tree_index]
        with open(f"{tree_dir}/{model_type}_tree_{tree_index}.txt", "w") as f:
            f.write(tree_text)
    
    elif model_type == 'LGBMRegressor':
        # LightGBM树可视化
        ax = plt.gca()
        lgb.plot_tree(model, tree_index=tree_index, ax=ax, show_info=['split_gain'])
        plt.title(f'LightGBM - 树 {tree_index}')
        
        # 保存文本表示
        tree_text = lgb.create_tree_digraph(model, tree_index=tree_index)
        with open(f"{tree_dir}/{model_type}_tree_{tree_index}.txt", "w") as f:
            f.write(tree_text)
    
    plt.tight_layout()
    plt.savefig(f"{tree_dir}/{model_type}_tree_{tree_index}.png", dpi=300)
    plt.close()
    print(f"已保存{model_type}的树结构可视化到 {tree_dir}")

def analyze_predictions(X, y_true, pipeline, model_type):
    """分析预测结果并识别最佳/最差拟合样本"""
    # 使用管道进行预测（自动预处理）
    y_pred = pipeline.predict(X)
    
    # 计算绝对误差
    errors = np.abs(y_true - y_pred)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        '真实价格': y_true,
        '预测价格': y_pred,
        '绝对误差': errors
    })
    
    # 合并原始特征用于分析
    results = pd.concat([X.reset_index(drop=True), results], axis=1)
    
    # 找出拟合最好的10个样本
    best_fit = results.nsmallest(10, '绝对误差')
    
    # 找出拟合最差的10个样本
    worst_fit = results.nlargest(10, '绝对误差')
    
    # 保存分析结果
    analysis_dir = f'../analysis/{model_type}'
    os.makedirs(analysis_dir, exist_ok=True)
    
    best_fit.to_csv(f"{analysis_dir}/{model_type}_best_fit.csv", index=False)
    worst_fit.to_csv(f"{analysis_dir}/{model_type}_worst_fit.csv", index=False)
    
    print(f"\n{model_type}模型分析结果:")
    print(f"拟合最好的样本平均绝对误差: {best_fit['绝对误差'].mean():.2f}")
    print(f"拟合最差的样本平均绝对误差: {worst_fit['绝对误差'].mean():.2f}")
    
    return best_fit, worst_fit

def plot_feature_importance(pipeline, feature_names, model_type):
    """绘制特征重要性图"""
    # 获取实际模型
    model = pipeline.named_steps['regressor']
    
    plt.figure(figsize=(12, 8))
    
    if model_type == 'RandomForestRegressor':
        importances = model.feature_importances_
    elif model_type == 'XGBRegressor':
        importances = model.feature_importances_
    elif model_type == 'LGBMRegressor':
        importances = model.feature_importances_
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    # 绘制水平条形图
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('特征重要性')
    plt.title(f'{model_type} - 特征重要性')
    plt.tight_layout()
    
    # 保存图像
    importance_dir = f'../feature_importance/{model_type}'
    os.makedirs(importance_dir, exist_ok=True)
    plt.savefig(f"{importance_dir}/{model_type}_feature_importance.png", dpi=300)
    plt.close()
    
    print(f"已保存{model_type}的特征重要性图")

def main():
    # 加载数据
    data = pd.read_csv('../data/US-pumpkins.csv')
    data = date_chuli0(data_rename(data))
    data = data_tezheng(data)
    data = pd.DataFrame(data)
    
    y = data['均价']
    X = data.drop(columns=['均价'])
    
    # 要分析的模型类型
    model_types = ['RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor']
    
    for model_type in model_types:
        try:
            print(f"\n{'='*60}")
            print(f"开始分析 {model_type} 模型")
            print('='*60)
            
            # 1. 加载模型
            pipeline = load_latest_model(model_type)
            print(f"成功加载 {model_type} 模型")
            
            # 2. 获取特征名称
            feature_names = get_feature_names(pipeline)
            print(f"获取到 {len(feature_names)} 个特征名称")
            
            # 3. 可视化并保存树结构
            actual_model = pipeline.named_steps['regressor']
            visualize_and_save_tree(actual_model, feature_names, model_type)
            
            # 4. 分析预测结果
            best_fit, worst_fit = analyze_predictions(X, y, pipeline, model_type)
            
            # 5. 绘制特征重要性
            plot_feature_importance(pipeline, feature_names, model_type)
            
            # 6. 分析最佳/最差拟合样本的特征分布
            print("\n拟合最好的样本特征分析:")
            print(best_fit[['城市名称', '包装', '品种', '产地', '物品尺寸', '真实价格', '预测价格']].describe(include='all'))
            
            print("\n拟合最差的样本特征分析:")
            print(worst_fit[['城市名称', '包装', '品种', '产地', '物品尺寸', '真实价格', '预测价格']].describe(include='all'))
            
        except Exception as e:
            print(f"处理 {model_type} 模型时出错: {e}")

if __name__ == '__main__':
    main()