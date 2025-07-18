"""
增强保存机制：确保所有可视化图表完整保存
"""

# 导入常用库
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

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

# 导入模型和评估工具
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 从自定义模块导入数据处理函数
from data_tezheng import pipeline_preprocessor_ColumnTransformer, date_chuli0, data_rename, data_tezheng

# 检查Graphviz是否可用
def is_graphviz_available():
    try:
        import graphviz
        test_graph = graphviz.Digraph()
        test_graph.node('A', 'Test')
        test_graph.render('test_graphviz', format='png', cleanup=True, quiet=True)
        return True
    except:
        return False

# 创建保存图表和结果的目录
def create_model_directories(model_name, timestamp):
    base_dir = f'../models/{model_name}_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    plots_dir = f'{base_dir}/plots'
    os.makedirs(plots_dir, exist_ok=True)
    return base_dir, plots_dir

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
    return [
        {
            'name': 'LinearRegression',
            'class': LinearRegression,
            'params': {},
            'is_tree_based': False,
            'model_type': 'linear'
        },
        {
            'name': 'RandomForestRegressor',
            'class': RandomForestRegressor,
            'params': {
                'max_depth': None, 
                'min_samples_leaf': 1, 
                'min_samples_split': 2, 
                'n_estimators': 50, 
                'n_jobs': 1
            },
            'is_tree_based': True,
            'model_type': 'sklearn_ensemble'
        },
        {
            'name': 'GradientBoostingRegressor',
            'class': GradientBoostingRegressor,
            'params': {
                'max_depth': None, 
                'min_samples_leaf': 1, 
                'min_samples_split': 2, 
                'n_estimators': 50
            },
            'is_tree_based': True,
            'model_type': 'sklearn_ensemble'
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
            },
            'is_tree_based': True,
            'model_type': 'xgboost'
        },
        {
            'name': 'LGBMRegressor',
            'class': LGBMRegressor,
            'params': {
                'n_estimators': 100, 
                'learning_rate': 0.1,
                'max_depth': 10,
                'force_col_wise': True,
                'verbose': -1,
                'n_jobs': 1
            },
            'is_tree_based': True,
            'model_type': 'lightgbm'
        }
    ]

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

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

# 保留：实际值vs预测值图
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

# 保留：特征重要性图
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

# 保留：树结构可视化图
def plot_tree_visualization(model, model_name, model_type, preprocessor, plots_dir, max_depth=2):
    # 检查Graphviz是否可用（XGBoost和LightGBM需要）
    graphviz_available = is_graphviz_available()
    if model_type in ['xgboost', 'lightgbm'] and not graphviz_available:
        print(f"未检测到Graphviz软件，跳过{model_name}树结构可视化")
        return
        
    feature_names = preprocessor.get_feature_names_out()
    
    if model_type == 'sklearn_ensemble':
        # sklearn集成模型的树可视化（不需要Graphviz）
        if isinstance(model, RandomForestRegressor) and hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            tree_to_plot = model.estimators_[0]
        elif isinstance(model, GradientBoostingRegressor) and hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            tree_to_plot = model.estimators_[0, 0]
        else:
            print(f"{model_name} 无可用决策树用于可视化")
            return
        
        plt.figure(figsize=(20, 12))
        plot_tree(
            tree_to_plot,
            max_depth=max_depth,
            filled=True,
            feature_names=feature_names,
            rounded=True,
            precision=2,
            fontsize=10
        )
        plt.title(f'{model_name} - 树结构可视化 (深度限制: {max_depth})')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_tree_visualization.png", bbox_inches='tight')
        plt.close()
        print(f"{model_name} 树结构可视化图已保存")
    
    elif model_type == 'xgboost' and isinstance(model, XGBRegressor):
        # XGBoost树可视化（需要Graphviz）
        tree_dump = model.get_booster().get_dump()[0]
        with open(f"{plots_dir}/{model_name}_tree_text.txt", "w") as f:
            f.write(tree_dump)
        
        plt.figure(figsize=(20, 12))
        ax = plt.gca()
        xgb.plot_tree(model, num_trees=0, ax=ax, rankdir='LR')
        plt.title(f'{model_name} - 树结构可视化 (树 0)')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_tree_visualization.png", bbox_inches='tight')
        plt.close()
        print(f"{model_name} 树结构可视化图已保存")
    
    elif model_type == 'lightgbm' and isinstance(model, LGBMRegressor):
        # LightGBM树可视化（需要Graphviz）
        tree_text = lgb.create_tree_digraph(model, tree_index=0, format='text')
        with open(f"{plots_dir}/{model_name}_tree_text.txt", "w") as f:
            f.write(str(tree_text))
        
        plt.figure(figsize=(20, 12))
        ax = plt.gca()
        lgb.plot_tree(model, tree_index=0, ax=ax, figsize=(20, 12))
        plt.title(f'{model_name} - 树结构可视化 (树 0)')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_tree_visualization.png", bbox_inches='tight')
        plt.close()
        print(f"{model_name} 树结构可视化图已保存")
    
    else:
        print(f"不支持的树模型类型: {model_type}")

# 增强：完整保存SHAP可视化（包括蜂群图、条形图、force plot）
def plot_shap_values(model, X_sample, preprocessor, model_name, plots_dir, is_tree_based):
    if not is_tree_based:
        return
        
    X_processed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    # 1. SHAP特征重要性条形图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar")
    plt.title(f'{model_name} - SHAP特征重要性')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{model_name}_shap_summary_bar.png")
    plt.close()
    print(f"SHAP特征重要性条形图已保存")
    
    # 2. SHAP蜂群图（展示特征值与SHAP值关系）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names)
    plt.title(f'{model_name} - SHAP值分布')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{model_name}_shap_summary_bee.png")
    plt.close()
    print(f"SHAP蜂群图已保存")
    
    # 3. 单个样本的force plot（解释单个预测）
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
        plt.savefig(f"{plots_dir}/{model_name}_shap_force_sample_{i}.png")
        plt.close()
        print(f"SHAP force plot (样本 {i}) 已保存")

# 保留：部分依赖图（辅助解释特征影响）
def plot_partial_dependence_plots(model, X, feature_names, model_name, plots_dir, is_tree_based):
    if not is_tree_based:
        return
        
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_n = min(3, len(importances))
        top_features = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(15, 5))
        PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X,
            features=top_features,
            feature_names=feature_names,
            ax=ax
        )
        plt.suptitle(f'{model_name} - 特征部分依赖图', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{model_name}_partial_dependence.png")
        plt.close()
        print(f"部分依赖图已保存")

def save_model_and_visualizations(pipeline, model_name, model_type, is_tree_based, 
                                 scores, fold_scores, X_test, y_test, y_pred, X_sample):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    # 如果是树模型，绘制树结构图、SHAP值和部分依赖图
    if is_tree_based:
        plot_tree_visualization(model, model_name, model_type, preprocessor, plots_dir, max_depth=2)
        plot_shap_values(model, X_sample, preprocessor, model_name, plots_dir, is_tree_based)
        
        X_processed = preprocessor.transform(X_test)
        plot_partial_dependence_plots(model, X_processed, 
                                     preprocessor.get_feature_names_out(), 
                                     model_name, plots_dir, is_tree_based)
    
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
    
    summary_path = "../models/model_summary_results.csv"
    results_df_summary = pd.DataFrame([results])
    if os.path.exists(summary_path):
        existing_results = pd.read_csv(summary_path)
        updated_results = pd.concat([existing_results, results_df_summary], ignore_index=True)
        updated_results.to_csv(summary_path, index=False)
    else:
        results_df_summary.to_csv(summary_path, index=False)
    
    # 保存每折详细结果
    if fold_scores is not None:
        fold_results_path = f"{base_dir}/{model_name}_{timestamp}_fold_results.csv"
        pd.DataFrame(fold_scores).to_csv(fold_results_path, index=False)
        print(f"详细折评估结果已保存到: {fold_results_path}")
    
    print(f"模型已保存到: {model_path}")
    print(f"汇总评估结果已保存到: {summary_path}")
    print(f"可视化图表已保存到: {plots_dir}")
    
    return model_path, base_dir

def run_all_list_models(X, y):
    model_configs = get_model_list()
    best_score = -float('inf')
    best_model_path = None
    best_model_name = None
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 为SHAP值计算准备样本数据
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    
    # 检查Graphviz是否可用并提示
    graphviz_status = "已安装" if is_graphviz_available() else "未安装"
    print(f"Graphviz状态: {graphviz_status}，{'支持树可视化' if is_graphviz_available() else '将跳过XGBoost/LightGBM树可视化'}")
    
    for config in model_configs:
        model_name = config['name']
        is_tree_based = config['is_tree_based']
        model_type = config['model_type']
        print(f"\n{'='*50}")
        print(f"模型: {model_name}")
        
        # 初始化模型
        model = config['class'](** config['params'])
        # 创建pipeline
        pipeline = model_pipeline(model)
            
        # 使用k折交叉验证评估
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        all_metrics = {'r2': [],'rmse': [],'mae': [],'mape': []}
        
        # 执行手动交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 训练模型
            pipeline.fit(X_fold_train, y_fold_train)
            # 预测并评估
            y_pred = pipeline.predict(X_fold_val)
            metrics = calculate_metrics(y_fold_val, y_pred)
            
            # 存储当前折的评估结果
            fold_metrics = {'fold': fold + 1}
            fold_metrics.update(metrics)
            fold_scores.append(fold_metrics)
            
            # 收集所有折的指标
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # 打印当前折结果
            print(f"折 {fold + 1}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        # 打印交叉验证汇总结果
        print(f"\n交叉验证结果:")
        print(f"R²: {np.mean(all_metrics['r2']):.4f} (±{np.std(all_metrics['r2']):.4f})")
        print(f"RMSE: {np.mean(all_metrics['rmse']):.4f} (±{np.std(all_metrics['rmse']):.4f})")
        print(f"MAE: {np.mean(all_metrics['mae']):.4f} (±{np.std(all_metrics['mae']):.4f})")
        print(f"MAPE: {np.mean(all_metrics['mape']):.2f}% (±{np.std(all_metrics['mape']):.2f}%)")
        
        # 在完整训练集上训练最终模型
        pipeline.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred)
        print(f"\n测试集评估结果:")
        print(f"R²: {test_metrics['r2']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.2f}%")
        
        # 保存模型、结果和可视化图表
        model_path, _ = save_model_and_visualizations(
            pipeline, model_name, model_type, is_tree_based, 
            all_metrics, fold_scores, X_test, y_test, y_pred, X_sample
        )
        
        # 更新最佳模型
        if test_metrics['r2'] > best_score:
            best_score = test_metrics['r2']
            best_model_path = model_path
            best_model_name = model_name
        
        print("="*50)
    
    # 输出最佳模型信息
    if best_model_path:
        print(f"\n最佳模型: {best_model_name}, R² = {best_score:.4f}")
        print(f"最佳模型路径: {best_model_path}")
        
        # 创建最佳模型的符号链接
        best_link_path = "../models/BEST_MODEL_LATEST.joblib"
        if os.path.exists(best_link_path):
            os.remove(best_link_path)
        try:
            os.symlink(os.path.abspath(best_model_path), best_link_path)
            print(f"已创建最佳模型的快捷方式: {best_link_path}")
        except:
            print(f"创建快捷方式失败，可手动访问最佳模型: {best_model_path}")
    
    return best_model_path

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
