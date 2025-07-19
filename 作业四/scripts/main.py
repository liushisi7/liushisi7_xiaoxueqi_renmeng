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
from show_jieguo_fx import *

# run_all_list_models和主函数
def run_all_list_models(X, y):
    model_configs = get_model_list()
    best_score = -float('inf')
    best_model_path = None
    best_model_name = None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    
    for config in model_configs:
        model_name = config['name']
        is_tree_based = config['is_tree_based']
        model_type = config['model_type']
        print(f"\n{'='*50}")
        print(f"模型: {model_name}")

        model = config['class'](** config['params'])
        pipeline = model_pipeline(model)
            
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        all_metrics = {'r2': [],'rmse': [],'mae': [],'mape': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pipeline.fit(X_fold_train, y_fold_train)
            y_pred = pipeline.predict(X_fold_val)
            metrics = calculate_metrics(y_fold_val, y_pred)
            
            fold_metrics = {'fold': fold + 1}
            fold_metrics.update(metrics)
            fold_scores.append(fold_metrics)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            print(f"折 {fold + 1}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        print(f"\n交叉验证结果:")
        print(f"R²: {np.mean(all_metrics['r2']):.4f} (±{np.std(all_metrics['r2']):.4f})")
        print(f"RMSE: {np.mean(all_metrics['rmse']):.4f} (±{np.std(all_metrics['rmse']):.4f})")
        print(f"MAE: {np.mean(all_metrics['mae']):.4f} (±{np.std(all_metrics['mae']):.4f})")
        print(f"MAPE: {np.mean(all_metrics['mape']):.2f}% (±{np.std(all_metrics['mape']):.2f}%)")
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred)
        print(f"\n测试集评估结果:")
        print(f"R²: {test_metrics['r2']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.2f}%")
        
        model_path, _ = save_model_and_visualizations(
            pipeline, model_name, model_type, is_tree_based, 
            all_metrics, fold_scores, X_test, y_test, y_pred, X_sample
        )
        
        if test_metrics['r2'] > best_score:
            best_score = test_metrics['r2']
            best_model_path = model_path
            best_model_name = model_name
        
        print("="*50)
    
    if best_model_path:
        print(f"\n最佳模型: {best_model_name}, R² = {best_score:.4f}")
        print(f"最佳模型路径: {best_model_path}")

    return best_model_path

def main():
    import os
    os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin/"
    
    shap.initjs()
    data = pd.read_csv('../data/US-pumpkins.csv')
    data = date_chuli0(data_rename(data))
    data = data_tezheng(data)
    data = pd.DataFrame(data)
    y = data['均价']
    X = data.drop(columns=['均价'])
    run_all_list_models(X, y)



if __name__ == '__main__':
    main()