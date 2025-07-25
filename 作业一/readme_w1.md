# 机器学习中的聚类模型

聚类（clustering）是一项机器学习任务，用于寻找类似对象并将他们分成不同的组（这些组称做“聚类”（cluster））。聚类与其它机器学习方法的不同之处在于聚类是自动进行的。事实上，我们可以说它是监督学习的对立面。

## 本节主题: 尼日利亚观众音乐品味的聚类模型🎧

尼日利亚多样化的观众有着多样化的音乐品味。使用从 Spotify 上抓取的数据，让我们看看尼日利亚流行的一些音乐。这个数据集包括关于各种歌曲的舞蹈性、声学、响度、言语、流行度和活力的分数。从这些数据中发现一些模式（pattern）会是很有趣的事情!

![A turntable](../images/turntable.jpg)

> <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a>在<a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>上的照片

使用聚类技术分析数据的新方法。当数据集缺少标签的时候，聚类特别有用。如果它有标签，那么分类技术可能会更有用。但是如果要对未标记的数据进行分组，聚类是发现模式的好方法。


## K-Means 聚类
### 原理
K-Means 聚类是一种无监督学习算法，其目标是将数据集中的样本划分为 K 个不同的簇（cluster）。算法的核心思想是通过迭代的方式，不断更新每个簇的中心点（质心，centroid），使得每个样本到其所属簇的中心点的距离之和最小。具体步骤如下：
1. **初始化**：随机选择 K 个样本作为初始的中心点。
2. **分配样本**：计算每个样本到各个中心点的距离，将样本分配到距离最近的中心点所在的簇。
3. **更新中心点**：对于每个簇，计算该簇内所有样本的均值，将其作为新的中心点。
4. **重复步骤 2 和 3**：直到中心点不再发生显著变化或达到最大迭代次数。

### 对值敏感
K-Means 聚类对数据的尺度比较敏感，因为它使用欧几里得距离来衡量样本之间的相似度。如果不同特征的尺度差异较大，那么尺度较大的特征在距离计算中会占据主导地位，从而影响聚类的结果。

### 归一化/标准化
为了避免尺度差异对聚类结果的影响，通常需要对数据进行归一化或标准化处理。归一化将数据缩放到 [0, 1] 区间，而标准化则将数据转换为均值为 0，标准差为 1 的分布。

### 高维数据适用性
K-Means 聚类在处理高维数据时可能会面临一些挑战，例如“维度灾难”问题。随着数据维度的增加，样本之间的距离会变得越来越相似，导致聚类效果变差。为了缓解这个问题，可以使用降维技术，如主成分分析（PCA）。

## 主成分分析（PCA）
### 原理
主成分分析是一种无监督学习的降维技术，其核心思想是通过线性变换将原始数据投影到一个新的低维空间，使得投影后的数据具有最大的方差。具体来说，PCA 会找到数据的主成分，即数据方差最大的方向，然后将数据投影到这些主成分上。

### 作用
- **数据降维**：减少数据的维度，降低计算复杂度，同时保留数据的主要信息。
- **去除噪声**：通过保留方差较大的主成分，去除方差较小的噪声成分。
- **可视化**：将高维数据降维到 2 维或 3 维，方便进行可视化分析。



### 实际业务分析
在尼日利亚观众音乐品味的聚类模型中，我们可以使用 K-Means 聚类算法对歌曲数据进行分组，从而发现不同音乐品味的群体。具体步骤如下：
1. **数据预处理**：对歌曲数据进行清洗、归一化或标准化处理。
2. **特征选择**：选择与音乐品味相关的特征，如 声学、响度等。
3. **降维（可选）**：如果数据维度较高，可以使用 PCA 进行降维处理。
4. **K-Means 聚类**：使用 K-Means 算法对数据进行聚类，确定不同的音乐品味群体。
5. **结果分析**：分析每个聚类的特征，了解不同音乐品味群体的喜好。

# 音乐风格聚类
![image](https://github.com/user-attachments/assets/22645674-7a41-4118-a7a6-fa9bf1b08d0d)
![image](https://github.com/user-attachments/assets/a1957c93-be4b-4f79-8908-a2b74fef6295)
### 一、聚类分布：类型A主导，小众类型特征鲜明  
饼图直观呈现音乐类型占比：  
- **类型A（聚类0）** 以 **361个样本、68.1%占比** 绝对主导，是平台最热门音乐类型基础盘，歌曲数量最多。  
- **类型D（聚类3）** 占21.1%，属第二梯队；类型B（聚类1，5.8%）、类型C（聚类2，4.9%）为小众类型，样本量少但特征独特，是差异化内容补充。  

这种分布反映平台音乐生态：大众偏好集中（类型A），同时存在细分需求（小众类型），运营可侧重“主流维稳 + 小众挖掘”。  


### 二、特征维度：四大特征刻画音乐类型差异  
结合**柱状图、热力图、聚类中心值**，从**响度、语音性、节拍速度、拍号** 4维度，拆解各类型核心特征：  

#### 1. 响度（特征0）：各类型差异小，类型A/B稍突出  
- 聚类0（类型A）特征值 **0.729**，聚类1（类型B）**0.740** ，均略高于均值（结合聚类分析，类型A“特征0高于均值0.007” 、类型B“特征0 0.740接近均值” ），说明两类音乐整体响度偏强，更易抓住用户听觉注意力（如流行音乐常靠响度营造氛围）。  
- 类型C（聚类2，0.702）、类型D（聚类3，0.700）响度稍弱，可能偏向轻音乐、氛围音乐等“低刺激”风格，满足用户放松、专注场景需求。  


#### 2. 语音性/视频成分（特征1）：小众类型“内容融合性”差异大  
- **类型D（聚类3）特征值0.480**，显著高于均值（“特征1高于平均值0.269” ），说明音乐中语音、视频元素占比高，可能是**影视OST、带念白的叙事音乐**，依赖内容联动增强沉浸感（如游戏配乐、广播剧插曲 ）。  
- **类型A（聚类0）特征值0.113**，远低于均值（“特征1低于平均值0.099” ），更接近“纯音乐”或“语音占比极低的歌曲”，适配短视频BGM、纯享听歌场景，通用性强。  
- 类型B（0.295）、类型C（0.329）处于中间值，语音/视频成分适中，可能是常规流行歌（含人声演唱，但不过度依赖旁白/剧情 ）。  


#### 3. 节拍速度（特征2）：类型B节奏最突出，类型D偏舒缓  
- **类型B（聚类1）特征值0.866**，大幅高于均值（“特征2高于平均值0.487” ），节奏快、律动感强，契合**舞曲、电子音乐** 场景，适合运动、派对等强节奏需求。  
- **类型D（聚类3）特征值0.322**，低于均值（“特征2低于平均值0.058” ），节奏偏慢，可能是民谣、古典音乐等“慢节奏叙事”风格，匹配休闲、冥想等低动态场景。  
- 类型A（0.353）、类型C（0.424）节奏中等，覆盖大众日常听歌习惯（如流行、摇滚常规节奏 ）。  


#### 4. 拍号（特征3）：类型C“节拍规则性”独特  
- **类型C（聚类2）特征值1.000**，远高于均值（“特征3高于平均值0.507” ），拍号特征极端化，可能是**古典音乐、爵士乐** 等对节拍结构要求高的类型（如古典乐多4/4、3/4拍，爵士乐常含复杂拍号变化 ），音乐专业性、艺术性突出。  
- **类型B（聚类1）特征值0.274**，低于均值（“特征3低于平均值0.219” ），拍号规则性弱，可能偏向自由节奏（如部分实验音乐、即兴创作 ），追求打破常规的听觉体验。  
- 类型A（0.488）、类型D（0.455）拍号特征常规，符合大众对“流行音乐节拍”的认知（如4/4拍主导 ）。  


### 三、聚类质量与应用：有区分度但需优化，可支撑内容运营  
- **聚类质量**：轮廓系数 **0.417** 说明聚类有一定区分度（值越接近1，类内相似度、类间差异度越高 ），但仍有优化空间（如小众类型样本量少，可能导致特征聚合不精准 ）。  
- **应用价值**：  
  - **内容推荐**：基于特征标签，给用户精准推送（如给“运动爱好者”推类型B快节奏音乐，给“专注办公”用户推类型D舒缓音乐 ）。  
  - **运营策略**：类型A主打“全民覆盖”（如首页banner、热门歌单 ）；小众类型（B/C/D）做“细分场景运营”（如专题企划、社群运营 ），挖掘垂直用户价值。  
  - **内容生产**：创作者可参考特征，针对性制作（如想做“影视联动音乐”，强化特征1；想做“舞曲”，强化特征2 ）。  


**总结**：该平台音乐类型呈“大众主导 + 小众细分”格局，4大特征清晰刻画类型差异。运营可围绕“主流维稳、小众深耕”策略，用特征标签优化推荐、内容生产，提升用户听歌匹配度与平台内容丰富度，后续可补充小众类型数据、优化聚类算法，进一步挖掘细分场景价值 。

# 歌手聚类和分析
![image](https://github.com/user-attachments/assets/2aa2726f-77f2-4063-968c-615ab42629cf)
![image](https://github.com/user-attachments/assets/ca2b1bc4-d4b2-4043-a3ec-e5559e3b1617)
![image](https://github.com/user-attachments/assets/5acbaf29-1f20-4642-addc-7e17fb852cf3)

### 一、聚类特征对比表（标准化值）
| 特征维度          | 聚类0       | 聚类1       | 聚类2       | 聚类3       |
|-------------------|-------------|-------------|-------------|-------------|
| 出道间隔时间(年)  | 0.502（高） | 0.102（低） | 0.250（中） | 0.256（中） |
| 最近发歌间隔时间(年) | 0.476（高） | 0.047（低） | 0.013（极低）| 0.218（中低）|
| 累计总热度        | 0.047（极低）| 0.085（低） | 0.670（高） | 0.050（极低）|
| 近三年发歌总数量  | 0.005（极低）| 0.225（中低）| 0.781（高） | 0.009（极低）|
| 近三年总热度      | 0.000（极低）| 0.157（中低）| 0.811（极高）| 0.003（极低）|

### 二、各聚类核心特征与业务定义
#### 1. **聚类0：沉寂型歌手（出道久且近期无活跃）**
- **特征**：  
  - 出道间隔时间（0.502）和最近发歌间隔（0.476）均为最高，累计热度（0.047）和近三年数据趋近于0。  
  - **典型案例**：如2Baba（最近发歌间隔7年，近三年无作品）。  
- **业务定位**：  
  - 历史老牌歌手，但长期未发歌，市场影响力基本消失。  
- **运营策略**：  
  - **风险预警**：优先评估续约价值，避免资源浪费；  
  - **激活可能**：若累计热度曾较高（如2Baba累计112），可尝试怀旧企划（如经典专辑重制）。

#### 2. **聚类1：新兴潜力型歌手（出道时间短，初步活跃）**
- **特征**：  
  - 出道间隔（0.102）和最近发歌间隔（0.047）最短，近三年发歌量（0.225）和热度（0.157）中等。  
  - **典型案例**：chike（出道1年，近三年发歌1首，热度30）。  
- **业务定位**：  
  - 刚入行的新人，处于市场试探期，有基础活跃度但影响力尚未爆发。  
- **运营策略**：  
  - **资源倾斜**：增加平台曝光（如推荐位、合辑收录），加速粉丝积累；  
  - **数据追踪**：重点监测后续作品热度变化，筛选潜力股。

#### 3. **聚类2：头部活跃型歌手（高热度+高产出）**
- **特征**：  
  - 近三年发歌量（0.781）和热度（0.811）均为最高，最近发歌间隔（0.013）极低，累计热度（0.670）高。  
  - **典型案例**：AYLØ（近三年发歌3首，热度72）、prettyboydo（发歌2首，热度52）。  
- **业务定位**：  
  - 当前市场的核心力量，高产出且高人气，处于职业生涯上升期。  
- **运营策略**：  
  - **重点投入**：签约独家合作、开发周边商业价值（如代言、巡演）；  
  - **风险防控**：避免过度消耗人气，合理规划发歌频率。

#### 4. **聚类3：边缘稳定型歌手（活跃度中等，热度低迷）**
- **特征**：  
  - 出道间隔（0.256）和最近发歌间隔（0.218）中等，但近三年发歌量（0.009）和热度（0.003）接近0。  
  - **典型案例**：Afro B（近三年发歌1首，热度0）。  
- **业务定位**：  
  - 出道时间不长不短，但作品影响力极弱，处于行业边缘。  
- **运营策略**：  
  - **成本控制**：减少推广投入，转为低成本试错（如短视频BGM合作）；  
  - **风格调整**：分析同簇歌手失败原因，探索转型可能性（如更换音乐类型）。


### 三、业务应用：资源分配与策略矩阵
#### 1. **四象限运营模型**
以**近三年总热度**为纵轴、**近三年发歌量**为横轴，各聚类分布如下：  
- **第一象限（高热度+高产出，聚类2）**：投入70%资源，打造标杆艺人；  
- **第二象限（高热度+低产出，无对应聚类）**：需挖掘潜在歌手；  
- **第三象限（低热度+低产出，聚类0/3）**：投入10%资源，或启动淘汰机制；  
- **第四象限（低热度+高产出，聚类1）**：投入20%资源，观察潜力。

#### 2. **合作优先级排序**
| 优先级 | 聚类       | 合作价值点                | 示例歌手       |
|--------|------------|---------------------------|----------------|
| 1      | 聚类2      | 高流量带动品牌曝光        | AYLØ、prettyboydo |
| 2      | 聚类1      | 低成本培育未来头部艺人    | chike           |
| 3      | 聚类0      | 怀旧IP开发（低成本激活）  | 2Baba           |
| 4      | 聚类3      | 谨慎合作或暂停投入        | Afro B         |

#### 3. **风险预警指标**
- **高风险簇**：聚类0（沉寂风险）、聚类3（边缘化风险），需监控发歌间隔超过2年的歌手；  
- **机会簇**：聚类1中近三年热度增速超过20%的歌手（如Adekunle Gold），可升级为重点培育对象。




