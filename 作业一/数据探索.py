# %% [markdown]
# # 尼日利亚观众音乐品味的聚类分析

# %%
# 数据分析常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 去除警告
import warnings
warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv('nigerian-songs.csv')
data

# %%
# 查看数据的基本信息
data.info()

# %% [markdown]
# 歌曲名称、专辑、艺术家、艺术家顶级流派、发行日期、长度、流行度、舞蹈性、音乐性、能量、乐器性、活跃度、响度、语速、节奏和时值  
# **各列都是530，均不存在空值，故无需进行缺失值处理。数据类型见Dtype。通过观察，这是一个比较规整的数据集，无需进行缺失值等处理。**
# 

# %%
# 替换列名称，便于理解
# 原始列名: ['name', 'album', 'artist', 'artist_top_genre', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature']
data.columns = ['歌曲名称', '专辑', '歌手', '歌手主要流派', '发行日期', '时长', '流行度', '舞蹈性', '声学性', '能量值', '器乐性', '现场感', '响度(dB)', '语音性/说唱成分', '节拍速度(BPM)', '拍号']
# 查看数据的前5行
data

# %%
# 统计每列的非重复值数量
data.nunique()

# %% [markdown]
# 

# %% [markdown]
# ##### 年度歌曲发布变化

# %%
data['发行日期'].value_counts().sort_index().plot(kind='line', figsize=(12, 3), color='skyblue',title='年度发行歌曲数量变化',xticks=range(min(data['发行日期']), max(data['发行日期']) + 1, 1),yticks=range(0, 100, 10),marker='o',grid=True,alpha=0.8  )
# print(data['发行日期'].value_counts().sort_index())

# %% [markdown]
# **发展特征：2008年后快速增长，从个位数跃升到两位数。2016年达到峰值（95首），是音乐创作最活跃的一年可能与数字音乐平台兴起相关。2014-2017年是黄金时期，308首约占占总量的58% 。此后逐渐下降。整体呈现先增长后近代逐渐下降的发展模式。**
# 

# %%
data['歌手主要流派'].value_counts().plot(kind='bar', figsize=(12, 3), color='skyblue',title='歌曲歌手主要流派出现频次分布')
# print(data['歌手主要流派'].value_counts())

# %% [markdown]
# **从上述可见对于尼日利亚歌曲数据集的中存在19个流派，且歌手主要流派偏好以afro dancehall流派最为绝对的头部流派占据328/497，其次的afropop 、Missing 、nigerian pop为中流派系整体（十位数量级），其他流派系占比较少，基本歌手数量低于10，属于小众流派。**

# %%
# data['歌手'].value_counts().head(15).plot(kind='bar',figsize=(12, 3),color='skyblue',title='歌手出现频次分布')

# %%
# top5的歌手的年度发行歌曲数量变化
gs_gq_y = data.groupby(['歌手', '发行日期']).size().reset_index(name='歌曲数量')
top_gesho = gs_gq_y[gs_gq_y['歌手'].isin(data['歌手'].value_counts().head(5).index)]
# 绘制图表-top5歌手年度发行歌曲数量变化
plt.figure(figsize=(12, 6))
sns.lineplot(data=top_gesho, x='发行日期', y='歌曲数量', hue='歌手', marker='o')
plt.title('Top5歌手年度发行歌曲数量变化')
plt.xlabel('发行日期')
plt.ylabel('歌曲数量')
# plt.xticks(rotation=45)
# plt.grid(True)
plt.show()
# print(top_gesho)

# %% [markdown]
# *从图中可以看到发哥数前五的歌手中，可能存在不同的情况：长青类如Various Artists-能长期持续的输出歌曲，重点新星如WizKid-短期出现发布大量歌曲，稳定型歌手如Flavour-不论市场情况如何均稳定输出。*    
# **后续可以继续对歌手进行分类，通过如，热度情况、发歌量、首歌年份、存续年份、等对歌手划分不同类别，帮助筛选当下有发展前景的歌手群体**

# %%
# 绘制年度-热度散点图
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='发行日期', y='流行度', hue='歌手', palette='viridis', legend=False)
plt.title('年度-热度散点图')
plt.xlabel('发行日期')
plt.ylabel('流行度')
plt.grid(True)
plt.show()

# %% [markdown]
# 1. 从图中可见随时间变化，歌曲整体热度上限和总量在不断持续提升，表明行业态势持续向好发展  
# 2. 猜测歌曲发行越早累计热度越高，但实际表明，这种关系并不明显，且反而是进今年的发行的歌曲热度较高，高热度且发行早的歌曲极少。
# 3. 2现象存在的原因可能是因为相关音乐平台兴起、大众接受度的提升、产业的成熟等，导致整体的提升

# %%
# top5流量热度的歌手
gs_ll_data = data.groupby(['歌手'])['流行度'].sum().reset_index()
gs_ll_data.sort_values(by='流行度', ascending=False).head(5)

# %%
data.describe()#查看数据的基本统计信息

# %%
# 相关性热力图
data1= data.drop(['歌曲名称', '专辑', '歌手', '歌手主要流派', '发行日期'], axis=1)
plt.figure(figsize=(12, 8))
sns.heatmap(data1.corr(method='pearson'), annot=True, cmap='coolwarm', fmt='.2f') #method='pearson'表示使用皮尔逊相关系数-皮尔逊相关系数是最常用的相关系数之一，适用于线性关系的度量
plt.title('相关性热力图')
plt.show()

# %% [markdown]
# 从热力图可见线性相关性最高响度(dB)与能量值为0.73，表明两者具有强的正相关性，往往**响度(dB)高的歌曲表现出的能量值也较高**。  
# 

# %%


# %%


# %% [markdown]
# ### 思考
# 1. 是否需要做分箱处理？
# 2. 是否需要做 归一化/标准化 处理？
# 3. 特征是否要进行降维？ 
# 4. 做上述和不做上述处理的影响是什么？


