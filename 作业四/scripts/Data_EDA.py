#导入常用库
import os
print(os.getcwd())
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
'''
:param <类属性名称>: <描述>
:type <类属性名称>: <类型>
:return: <返回值,对返回值的描述>
:rtype: <返回值类型>
:raise <异常名称>: <异常描述>
'''

from data_tezheng import date_chuli0,data_rename

def str_data_show(data, column_name):
    '''
    此函数用于对str类标签，离散数据进行统计和绘图，便于快速理解数据
    '''
    # 统计每个标签/子属性的数量分布 
    data[column_name].value_counts()
    pass


def data_show_gaikuang(data):
    '''
    数据探索性分析
    :param data: 待处理的数据
    '''
    print("="*99)
    # 查看数据情况
    print(data.info());print("="*99)
    # 检查缺失值
    print('存在缺失值的属性:\n',data.isnull().sum()[data.isnull().sum() > 0]);print("="*99)
    # 检查重复值
    print('检查重复值:\n',data.duplicated().sum());print("="*99)
    # 检查数据分布
    print('检查数据分布:\n',data.describe());print("="*99)
    

def auto_eda(data, column_name,n_shu=15):
    '''
    对数据列进行自动检测并执行适当的探索性数据分析(EDA)
    参数:
    data: pandas DataFrame 数据框
    column_name: 要分析的列名
    n_shu: 选择后续绘图分析展示的前多少个数值型特征数量，如果少于该数量，则全部展示
    返回:
    stats_dict: 包含该列统计信息的字典
    '''
    print("="*130)
    print("="*50,f"{column_name}的自动探索性数据分析","="*50)
    print("="*130)
    if column_name not in data.columns:
        print(f"列名 '{column_name}' 不在数据集中")
    
    if data[column_name] is None or data[column_name].isnull().all():
        print(f"⚠⚠⚠⚠列名 '{column_name}' 中数据为空!⚠⚠⚠⚠")
        return None

    
    # 创建一个字典来存储统计信息
    stats_dict = {}
    
    # 获取列数据
    col_data = data[column_name]
    
    # 获取数据类型
    dtype_name = str(col_data.dtype)
    stats_dict['数据类型'] = dtype_name
    
    # 检查缺失值
    missing_count = col_data.isnull().sum()
    missing_percent = missing_count / len(col_data) * 100
    stats_dict['缺失值数量'] = missing_count
    stats_dict['缺失值百分比'] = f"{missing_percent:.2f}%"
    
    # 检查唯一值数量
    unique_count = col_data.nunique()
    stats_dict['唯一值数量'] = unique_count
    plt.figure(figsize=(12, 8))
    
    # 处理数值型数据
    if pd.api.types.is_numeric_dtype(col_data):
        print(f"检测到数值型数据: {column_name}")
        # 基本统计量
        stats = col_data.describe()
        stats_dict['统计描述'] = stats
        print(f"基本统计量:\n{stats}")
        
        # 创建子图布局
        plt.subplot(2, 2, 1)
        # 直方图
        sns.histplot(col_data.dropna(), kde=True)
        plt.title(f"{column_name} 的分布")
        plt.xlabel(column_name)
        plt.ylabel("频率")
        
        plt.subplot(2, 2, 2)
        # 箱线图
        sns.boxplot(y=col_data.dropna())
        plt.title(f"{column_name} 的箱线图")
        plt.ylabel(column_name)
        
        plt.subplot(2, 2, 3)
        # QQ图检验正态性
        from scipy import stats as scistat
        qq = scistat.probplot(col_data.dropna(), dist="norm", plot=plt)
        plt.title(f"{column_name} 的QQ图")
        
        plt.subplot(2, 2, 4)
        # 小提琴图
        sns.violinplot(y=col_data.dropna())
        plt.title(f"{column_name} 的小提琴图")
        plt.ylabel(column_name)
        
    # 处理分类/字符串数据
    elif pd.api.types.is_string_dtype(col_data) or unique_count < len(col_data) * 0.5:
        print(f"检测到分类/字符串数据: {column_name}")
        print(f'{column_name}的属性有{ unique_count }个')
        
        # 如果唯一值太多，只绘制前N个
        N = min(n_shu, unique_count)
        if N < unique_count:
            print(f"⚠⚠注意该特征属性太多，只绘制前 {N} 个！！！")

        # 计算频率
        value_counts = col_data.value_counts()
        value_percent = col_data.value_counts(normalize=True) * 100
        stats_dict['值计数'] = value_counts
        stats_dict['值百分比'] = value_percent
        
        print(f"前{N}个频率值:\n{value_counts.head(N)}")
        print(f"前{N}个百分比:\n{value_percent.head(N).apply(lambda x: f'{x:.2f}%')}")
        

        
        plt.subplot(2, 1, 1)
        # 条形图
        sns.countplot(y=col_data, order=value_counts.index[:N])
        plt.title(f"{column_name} 的前 {N} 个值分布")
        plt.ylabel(column_name)
        plt.xlabel("计数")
        
        plt.subplot(2, 1, 2)
        # 饼图
        plt.pie(value_counts.head(N), labels=value_counts.index[:N], autopct='%1.1f%%')
        plt.title(f"{column_name} 的前 {N} 个值占比")
    
    # 处理日期时间数据
    elif pd.api.types.is_datetime64_dtype(col_data):
        print(f"检测到日期时间数据: {column_name}")
        
        stats_dict['最早日期'] = col_data.min()
        stats_dict['最晚日期'] = col_data.max()
        stats_dict['时间跨度'] = col_data.max() - col_data.min()
        
        plt.subplot(2, 1, 1)
        # 按年月统计
        time_series = pd.Series(np.ones(len(col_data)), index=col_data)
        monthly = time_series.resample('M').count()
        monthly.plot(kind='line')
        plt.title(f"{column_name} 按月分布")
        
        plt.subplot(2, 1, 2)
        # 按周几统计
        day_of_week = col_data.dt.day_name().value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        sns.barplot(x=day_of_week.index, y=day_of_week.values)
        plt.title(f"{column_name} 按星期几分布")
        plt.xticks(rotation=45)
    
    # 处理布尔数据
    elif pd.api.types.is_bool_dtype(col_data):
        print(f"检测到布尔数据: {column_name}")
        
        value_counts = col_data.value_counts()
        stats_dict['值计数'] = value_counts
        value_percent = col_data.value_counts(normalize=True) * 100
        stats_dict['值百分比'] = value_percent
        
        plt.subplot(1, 2, 1)
        # 条形图
        sns.countplot(x=col_data)
        plt.title(f"{column_name} 的分布")
        plt.xlabel(column_name)
        
        plt.subplot(1, 2, 2)
        # 饼图
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f"{column_name} 的占比")
    
    # 其他数据类型
    else:
        print(f"其他类型的数据: {column_name}, 类型: {dtype_name}")
        
        # 尝试计算频率
        try:
            value_counts = col_data.value_counts().head(N)
            stats_dict[f'前{N}个值计数'] = value_counts
            print(f"前{N}个频率值:\n{value_counts}")
            
            plt.subplot(1, 1, 1)
            # 简单条形图
            sns.countplot(y=col_data, order=value_counts.index)
            plt.title(f"{column_name} 的前{N}个值分布")
            plt.ylabel(column_name)
        except:
            print(f"无法为列 {column_name} 生成可视化")
    
    plt.tight_layout()
    plt.show()

    return stats_dict


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

def data_y_xi_show(data, x, y):
    """
    y为数值类型，x为数值类型或分类类型或者日期时间类型
    此函数通过自动检测数据类型，并通过散点图+均值折线可视化展示x与y之间的关系
    
    参数:
    data: pandas DataFrame 数据框
    x: 字符串，自变量列名
    y: 字符串，因变量列名（必须是数值类型）
    
    注：当x为数值或者日期时间类型时，此函数会自动根据x大小排序绘制折线；
        当x为分类类型时，此函数会自动根据x的y均值排序绘制折线
    """
    # 检查列是否存在
    if x not in data.columns:
        raise ValueError(f"列 '{x}' 不在数据框中")
    if y not in data.columns:
        raise ValueError(f"列 '{y}' 不在数据框中")
    
    # 检查y是否为数值型
    if not pd.api.types.is_numeric_dtype(data[y]):
        raise TypeError(f"y列 '{y}' 必须是数值类型")
    
    if x not in data.columns:
        print(f"列名 '{x}' 不在数据集中")
    
    if data[x] is None or data[x].isnull().all():
        print(f"⚠⚠⚠⚠列名 '{x}' 中数据为空!⚠⚠⚠⚠")
        return None
    
    # 创建一个副本避免修改原始数据
    df = data.copy()
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 判断x的数据类型
    if pd.api.types.is_numeric_dtype(df[x]):
        # 数值型x
        print(f"检测到数值型x: {x}")
        
        # 排序数据，以便正确绘制折线图
        sorted_df = df.sort_values(by=x)
        
        # 数值分箱处理
        n_bins = min(20, len(df[x].unique()))  # 限制最大分箱数
        
        # 创建分箱
        bins = np.linspace(df[x].min(), df[x].max(), n_bins + 1)
        bin_labels = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        df['bin'] = pd.cut(df[x], bins=bins, labels=bin_labels)
        
        # 计算每个分箱的均值
        bin_avg = df.groupby('bin')[y].mean().reset_index()
        
        # 散点图展示原始数据
        plt.scatter(df[x], df[y], alpha=0.5, label='原始数据点')
        
        # 绘制均值折线
        plt.plot(bin_avg['bin'], bin_avg[y], 'r-o', linewidth=2, label=f'{x}分组平均{y}')
        
        # 绘制全局均值线
        plt.axhline(y=df[y].mean(), color='g', linestyle='--', label=f'全局平均{y}')
        
        plt.title(f'{x}与{y}的关系', fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
    elif pd.api.types.is_datetime64_dtype(df[x]) or isinstance(df[x].iloc[0], (datetime, pd.Timestamp)):
        # 日期时间型x
        print(f"检测到日期时间型x: {x}")
        
        # 确保x是日期时间类型
        if not pd.api.types.is_datetime64_dtype(df[x]):
            df[x] = pd.to_datetime(df[x])
        
        # 排序数据，以便正确绘制折线图
        sorted_df = df.sort_values(by=x)
        
        # 计算每天的均值
        daily_avg = df.groupby(x)[y].mean().reset_index()
        
        # 散点图展示原始数据
        plt.scatter(df[x], df[y], alpha=0.5, label='原始数据点')
        
        # 绘制均值折线
        plt.plot(daily_avg[x], daily_avg[y], 'r-o', linewidth=2, label=f'日期平均{y}')
        
        # 绘制全局均值线
        plt.axhline(y=df[y].mean(), color='g', linestyle='--', label=f'全局平均{y}')
        
        plt.title(f'{x}与{y}的关系', fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
    else:
        # 分类型x
        print(f"检测到分类型x: {x}")
        
        # 限制类别数量
        max_categories = 15
        top_categories = df[x].value_counts().nlargest(max_categories).index.tolist()
        
        filtered_df = df
        
        # 计算每个类别的y均值并按均值降序排序
        cat_avg = filtered_df.groupby(x)[y].mean().sort_values(ascending=False)
        cat_order = cat_avg.index
        
        # 创建映射字典，用于对原始数据中的类别进行排序
        cat_map = {cat: i for i, cat in enumerate(cat_order)}
        
        # 获取排序后的位置
        cat_positions = [cat_map.get(c, -1) for c in filtered_df[x]]
        
        # 散点图展示原始数据（按均值排序后的位置）
        plt.scatter(cat_positions, filtered_df[y], alpha=0.5, label='原始数据点')
        
        # 绘制均值折线
        positions = np.arange(len(cat_avg))
        plt.plot(positions, cat_avg.values, 'r-o', linewidth=2, label=f'{x}类别平均{y}')
        
        # 绘制全局均值线
        plt.axhline(y=df[y].mean(), color='g', linestyle='--', label=f'全局平均{y}')
        
        plt.title(f'{x}与{y}的关系（按{y}降序排序' + 
                 (f'，仅显示前{max_categories}种）' if len(df[x].unique()) > max_categories else '）'), 
                 fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.xticks(range(len(cat_order)), cat_order, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
    
    # 适应图形布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # 返回一些统计信息
    result = {
        "相关系数": df[[x, y]].corr().iloc[0, 1] if pd.api.types.is_numeric_dtype(df[x]) else None,
        f"{y}均值": df[y].mean(),
        f"{y}中位数": df[y].median(),
        f"{y}标准差": df[y].std(),
        f"{y}最小值": df[y].min(),
        f"{y}最大值": df[y].max()
    }
    
    # 如果x是分类型，返回每个类别的y均值（降序排列）
    if not (pd.api.types.is_numeric_dtype(df[x]) or pd.api.types.is_datetime64_dtype(df[x])):
        result[f"按{y}降序排列的{x}均值"] = df.groupby(x)[y].mean().sort_values(ascending=False).to_dict()
    
    return result





def data_show_wenti(data):
    """
    此函数用于回答自定义提出的四个问题
    1.各州南瓜产量分析  2.南瓜价格最高地区分析  3.南瓜尺寸与价格关系分析  4.南瓜品种价格分析  5.月份与价格/品种数量关系分析
    同时并给出简单的分析
    *注：此函数仅仅使用本数据集，不通用
    """
    print("=== 问题1: 各州南瓜产量分析 ===")
    # 按产地统计产量
    origin_production = data['产地'].value_counts().reset_index(name='产量')
    origin_production = origin_production.sort_values('产量', ascending=False)
    print("各州南瓜产量排名:")
    print(origin_production)

    # 各产地的物品尺寸分布
    origin_size_dist = data.groupby(['产地', '物品尺寸'])['物品尺寸'].count().unstack(fill_value=0)
    print("\n各产地南瓜尺寸分布:")
    print(origin_size_dist)

    # 各产地的品种分布
    origin_variety_dist = data.groupby(['产地', '品种'])['品种'].count().unstack(fill_value=0)
    print("\n各产地南瓜品种分布:")
    print(origin_variety_dist)

    # 可视化产量排名
    plt.figure(figsize=(10, 6))
    plt.bar(origin_production['产地'], origin_production['产量'], color='skyblue')
    plt.title('各州南瓜产量对比')
    plt.xlabel('产地')
    plt.ylabel('产量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("\n=== 问题2: 南瓜价格最高地区分析 ===")
    # 按城市计算均价
    city_price = data.groupby('城市名称')['均价'].mean().reset_index()
    city_price = city_price.sort_values('均价', ascending=False)
    print("各城市南瓜均价:")
    print(city_price)

    # 按产地计算均价
    origin_price = data.groupby('产地')['均价'].mean().reset_index()
    origin_price = origin_price.sort_values('均价', ascending=False)
    print("\n各产地南瓜均价:")
    print(origin_price)

    # 可视化价格最高的城市和产地
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(city_price.head(5)['城市名称'], city_price.head(5)['均价'], color='salmon')
    plt.title('价格最高的5个城市')
    plt.xlabel('城市')
    plt.ylabel('均价')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(origin_price.head(5)['产地'], origin_price.head(5)['均价'], color='lightgreen')
    plt.title('价格最高的5个产地')
    plt.xlabel('产地')
    plt.ylabel('均价')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("\n=== 问题3: 南瓜尺寸与价格关系分析 ===")
    # 按尺寸计算均价
    size_price = data.groupby('物品尺寸')['均价'].agg(['mean', 'count']).reset_index()
    size_price = size_price.sort_values('mean', ascending=False)
    print("不同尺寸南瓜的均价:")
    print(size_price)

    # 可视化尺寸与价格关系
    plt.figure(figsize=(8, 6))
    plt.bar(size_price['物品尺寸'], size_price['mean'], color='purple')
    plt.title('南瓜尺寸与均价关系')
    plt.xlabel('物品尺寸')
    plt.ylabel('平均价格')
    plt.tight_layout()

    plt.show()
    # 问题4: 哪种南瓜最贵？最便宜？
    print("\n=== 问题4: 南瓜品种价格分析 ===")
    # 按品种计算均价
    variety_price = data.groupby('品种')['均价'].agg(['mean', 'count']).reset_index()
    variety_price = variety_price.sort_values('mean', ascending=False)
    print("各品种南瓜均价:")
    print(variety_price)

    # 找出最贵和最便宜的品种
    most_expensive = variety_price.iloc[0]
    cheapest = variety_price.iloc[-1]
    print(f"\n最贵的品种: {most_expensive['品种']}, 均价: {most_expensive['mean']:.2f}")
    print(f"最便宜的品种: {cheapest['品种']}, 均价: {cheapest['mean']:.2f}")

    # 可视化品种价格
    plt.figure(figsize=(10, 6))
    plt.bar(variety_price['品种'], variety_price['mean'], color='orange')
    plt.title('各品种南瓜均价对比')
    plt.xlabel('品种')
    plt.ylabel('平均价格')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.savefig('variety_price_comparison.png')
    plt.show()
    print("\n=== 问题5: 月份与价格/品种数量关系分析 ===")
    # 按月份计算均价
    monthly_price = data.groupby('月份')['均价'].mean().reset_index()
    print("各月份南瓜均价:")
    print(monthly_price)

    # 按月份统计各品种数量
    monthly_variety_count = data.groupby(['月份', '品种'])['品种'].count().unstack(fill_value=0)
    print("\n各月份南瓜品种数量分布:")
    print(monthly_variety_count)

    # 可视化价格随月份变化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(monthly_price['月份'], monthly_price['均价'], marker='o', color='blue')
    plt.title('各月份南瓜均价变化')
    plt.xlabel('月份')
    plt.ylabel('平均价格')
    plt.xticks(monthly_price['月份'])

    # 可视化各月份品种数量变化
    plt.subplot(1, 2, 2)
    bottom = np.zeros(len(monthly_variety_count.index))
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i, variety in enumerate(monthly_variety_count.columns):
        plt.bar(monthly_variety_count.index, monthly_variety_count[variety], 
                bottom=bottom, label=variety, color=colors[i % len(colors)])
        bottom += monthly_variety_count[variety].values
    plt.title('各月份南瓜品种数量变化')
    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('monthly_analysis.png')
    plt.show()
    # 综合分析结果输出
    print("\n=== 综合分析结论 ===")
    print(f"1. 产量最多的州: {origin_production.iloc[0]['产地']}，产量为{origin_production.iloc[0]['产量']}")
    print(f"2. 价格最高的城市: {city_price.iloc[0]['城市名称']}，均价为{city_price.iloc[0]['均价']:.2f}")
    print(f"3. 最贵的南瓜尺寸: {size_price.iloc[0]['物品尺寸']}，均价为{size_price.iloc[0]['mean']:.2f}")
    print(f"4. 最贵的南瓜品种: {most_expensive['品种']}，均价为{most_expensive['mean']:.2f}")
    print(f"5. 价格最高的月份: {monthly_price.iloc[monthly_price['均价'].idxmax()]['月份']}月，均价为{monthly_price['均价'].max():.2f}")





if __name__ == '__main__':
    # 读取数据
    data =pd.read_csv(r'../data/US-pumpkins.csv')
    data = date_chuli0(data_rename(data))
    data_show_gaikuang(data)
    # data_show_wenti(data)

    for i in data.columns:
        # auto_eda(data,i,n_shu=14)
        data_y_xi_show(data,i,'均价')
    # print(data.head())
    pass

