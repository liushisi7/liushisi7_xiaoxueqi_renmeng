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


def data_rename(data):
    '''
    重新命名列名，帮助我这个不会英语的人更好理解数据
    :param data: 待处理的数据
    :return: 重新命名后的数据'''
    # 检查列名
    # print(data.columns.tolist())
    # 创建英文列名到中文列名的映射字典
    column_mapping = {'City Name': '城市名称','Type': '类型','Package': '包装','Variety': '品种','Sub Variety': '子品种','Grade': '等级','Date': '日期','Low Price': '最低价格','High Price': '最高价格','Mostly Low': '主要最低价','Mostly High': '主要最高价', 'Origin': '产地','Origin District': '产地区域','Item Size': '物品尺寸','Color': '颜色','Environment': '环境','Unit of Sale': '销售单位','Quality': '质量','Condition': '状况','Appearance': '外观','Storage': '储存','Crop': '作物','Repack': '重新包装','Trans Mode': '运输模式','Unnamed: 24': '未命名: 24','Unnamed: 25': '未命名: 25'}
    # 替换DataFrame的列名
    data.rename(columns=column_mapping, inplace=True)
    return data
def date_chuli0(data):
    '''
    将日期列转换为日期格式
    :param data:
    :return:日期格式化后的数据
    '''
    # data_rename(data)
    # 将日期列转换为日期格式
    data['日期'] = pd.to_datetime(data['日期'], format='%m/%d/%y')
    data["年份"] = data['日期'].dt.year
    data["月份"] = data['日期'].dt.month
    data["日"] = data['日期'].dt.day
    data["星期"] = data['日期'].dt.weekday
    data.drop(columns=['日期'], inplace=True)
    data['均价']= (data['最低价格'] + data['最高价格']) / 2
    return data

def data_Hot_Deck_Imputation(data):
    """
    热卡填补法-手动查找
    """
    print("="*50,'热卡填补法-手动',"="*50)
    print('检查填充前空值情况：',data['产地'].isnull().sum()) 
    # 使用空值记录中其他存在的特征相同的记录的品种 填充，如 城市名称、包装、产地、（颜色、尺寸）------ 使用其中出现频率最高的品种属性填充
    data['产地'].fillna(data.groupby(['城市名称','包装','物品尺寸','品种'])['产地'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
    print('检查填充后是否还有空值：',data['产地'].isnull().sum())
    print('检查填充前空值情况：',data['品种'].isnull().sum()) 
    # 使用空值记录中其他存在的特征相同的记录的品种 填充，如 城市名称、包装、产地、（颜色、尺寸）------ 使用其中出现频率最高的品种属性填充
    data['品种'].fillna(data.groupby(['城市名称','包装','物品尺寸','产地','最低价格'])['品种'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
    print('检查填充后是否还有空值：',data['品种'].isnull().sum()) # 检查填充后是否还有空值
    data['品种'].fillna(data.groupby(['城市名称','包装','物品尺寸','产地'])['品种'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
    print('放宽填要求后是否还有空值：',data['品种'].isnull().sum()) # 品牌成功填充
    print('='*110)
    return data

def data_tezheng(data):
    """
    数据特征分析和特征处理以及选择
    data: 数据集
    return: 特征处理后的,仅仅包含所需要的特征x和目标y的数据集
    *注意在时间维度上，去除日期会导致相似/相同 记录的重复，可能会造成数据泄露---需要对y进行处理，去除重复的y（删除/均值）
    """
    data = data_Hot_Deck_Imputation(data) # 热卡填补法
    # 重新包装中N与E的数量比过大，E只有5条记录，且存在至少两个特征的缺失值，故不考虑重新包装特征
    data.drop(columns=['重新包装'], inplace=True)
    # 删除全部为空的列
    data = data.dropna(axis=1, how='all')
    # 删除空值数量过多的列---该类数据的信息量太少，对后续分析无意义
    data = data.dropna(thresh=len(data)*0.2, axis=1) #thresh参数:指定保留的行或列中至少应包含的非缺失值的数量(可以容忍的缺失值数量)
    data['均价']= (data['最低价格'] + data['最高价格']) / 2

    # 去除异常值
    q1 = data['均价'].quantile(0.25)
    q3 = data['均价'].quantile(0.75)
    iqr = q3 - q1
    data = data[~((data['均价'] < (q1 - 1.5 * iqr)) | (data['均价'] > (q3 + 1.5 * iqr)))]
    data=data[[ '月份', '星期', '物品尺寸', '包装', '品种', '产地','城市名称','均价']]
    # 去除其他无法填充，依旧存在空值的记录
    data = data.dropna(axis=0, how='any')
    # !!!!! 对重复的记录进行处理，去除重复的记录，保留一条记录（此处取均值）!!!!!----使用后数据量大幅度下降
    # data=data.groupby(['月份', '星期', '物品尺寸', '包装', '品种', '产地','城市名称'], as_index=False).mean()
    # y=data['均价']
    # x=data.drop(columns=['均价'])
    # print(data.info())
    # print('数据集大小：',data.shape)


    return data


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
def pipeline_preprocessor_ColumnTransformer(numerical_features=None, nominal_features=None, ordinal_features=None, ordinal_features_zd=None):
    """
    创建列转换器-后续可用直接在管道中加入,其中
    numerical_features: 数值特征-示例：numerical_features=['数值特征1', '数值特征2']
    nominal_features: 名义分类特征-示例：nominal_features=['名义分类特征1', '名义分类特征2']  
    ordinal_features: 有序分类特征-示例：ordinal_features=['有序分类特征1', '有序分类特征2']
    ordinal_features_zd: 有序分类特征的字典-示例：ordinal_features_zd={'特征1':['小','中','大'], '特征2':[ '低', '中','高']}
    编码输出中名义编码随机分配，有序编码按照字典顺序编码
    """
    transformers_list=[]
    if numerical_features:
        transformers_list.append(('num', StandardScaler(), numerical_features))
    if nominal_features:
        transformers_list.append(('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal_features))
    if ordinal_features and ordinal_features_zd:
        transformers_list.append(('ord', OrdinalEncoder(categories=[ordinal_features_zd[feature] for feature in ordinal_features]), ordinal_features))
    # 创建列转换器-后续可用直接在管道中加入
    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        # remainder='drop'
        remainder='passthrough'
    )
    
    return preprocessor



if __name__ == '__main__':
    # 读取数据集
    data = pd.read_csv('../data/US-pumpkins.csv')
    data=date_chuli0(data_rename(data))
    # 数据特征分析和特征处理以及选择
    data = data_tezheng(data)
    data=pd.DataFrame(data)
    # print(data['物品尺寸'].value_counts())

    #=========================================================================================================================
    # 测试pipeline_preprocessor_ColumnTransformer
    # numerical_features = ['均价']
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
    
    # 应用预处理器转换数据
    transformed_data = preprocessor.fit_transform(data)

    # 创建转换后数据的列名
    transformed_columns = []
    # 添加数值特征的列名
    # if numerical_features:
    #     transformed_columns.extend(numerical_features)
    # 添加名义特征的One-Hot编码列名
    if nominal_features:
        # 获取OneHotEncoder的输出特征名
        nom_encoder = preprocessor.named_transformers_['nom']
        nom_feature_names = nom_encoder.get_feature_names_out(nominal_features)
        transformed_columns.extend(nom_feature_names)

    # 添加有序特征的列名
    if ordinal_features:
        transformed_columns.extend(ordinal_features)
        
    # 添加剩余的未转换特征列名（处理remainder='passthrough'）
    # 获取所有特征列名
    all_features = data.columns.tolist()
    # 获取已经被转换的特征列名
    transformed_features = []
    if nominal_features:
        transformed_features.extend(nominal_features)
    if ordinal_features:
        transformed_features.extend(ordinal_features)
    # 找出未被转换的特征列名
    passthrough_features = [col for col in all_features if col not in transformed_features]
    # 添加到转换后的列名列表
    transformed_columns.extend(passthrough_features)

    # 验证列名数量与数据形状是否匹配
    print(f"数据形状: {transformed_data.shape}")
    print(f"列名数量: {len(transformed_columns)}")
    
    # 如果列名数量与数据列数不匹配，则使用简单的列序号
    if transformed_data.shape[1] != len(transformed_columns):
        print("警告：列名数量与数据形状不匹配，使用序号作为列名")
        transformed_columns = [f'feature_{i}' for i in range(transformed_data.shape[1])]
    
    # 创建包含转换后数据的DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
    print(transformed_df.head())


    
