---
title: "什么样的CEO更优秀？横跨四大洲1114位CEO行为数据的研究精析"
date: 2026-11-12T13:10:00+08:00
draft: false
summary: "CEO的行为模式能否被客观量化？能否据此判断哪类CEO更适合哪类企业？本文拆解《CEO Behavior and Firm Performance》，从 1114 位 CEO 的 4253 项行为记录出发，用 LDA 把行为压缩成一个指数，并完整复现其建模过程。"
showToc: true
tocOpen: false
tags: ["CEO", "组织经济学", "LDA", "论文拆解"]
categories: ["研究"]
---

CEO行为与企业绩效之间的关系是组织经济学和管理学的重要研究问题。**CEO的行为模式如何影响企业绩效？** 是否能根据行为模式将CEO分为不同类型？而不同类型的CEO是否适合不同类型的企业？

但是「行为模式」大多根据人们在日常工作中的经验观察，同一个CEO面对不同的人行为可能大不相同，比如有网友总结道：

> 中国的两大互联网巨头——腾讯和阿里，创始人的风格非常不同。在公众面前，马云的形象是高谈阔论，而马化腾则显得较为低调。在公司管理上，马云不插手具体事务，而是站在高处务虚，抓战略、抓文化，而马化腾则是腾讯的“超级产品经理”，亲自加入到不同产品的战斗中去。

**然而我们并不知道这种行为模式分类到底多大程度上是“真实”的，因此这成为一个难以量化、无法研究的主观问题**。

2020年，来自哈佛商学院，牛津大学，哥伦比亚大学和伦敦政经学院的四位研究人员用数据与机器学习的方法为这个问题提供了漂亮的解决方案：他们找到了一种方法来客观量化CEO的行为风格，并基于横跨四大洲、六个国家的**1000+位CEO**行为数据，找到了行为风格与公司表现之间的普遍关系。这就是本文介绍的研究《CEO Behavior and Firm Performance》。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20260928151948186.png)

笔者硕士期间有幸是作者之一的学生，用原数据集复现过该论文，收获很大。在这里对本文做全方位细致的拆解，将核心结论、建模思路分享出来，希望不仅对同行有所帮助，也兼有科普性，帮外行增长认知。



## 1 研究简述

### 1\.1 背景、挑战与创新的解决方法

**学界感兴趣但难以研究的问题**：CEO在公司运营中都做些什么，以及这些行为如何导致公司表现的不同。

- **挑战1**: 如何收集一大批CEO的数据？**解决方案**：通过他们结构化的日志数据 (而不是他们本人)  - 因此具备了 "可比较性 \& 客观性"
    - 1114 CEOs,  每个CEO记录50个工作小时, 共4253 个行为
    - 但这带来 **第二个挑战: 过于高维的信息财富**
- **挑战2**: 如何将CEOs活动的细粒度信息聚合成一个跨主题具有一致意义的摘要度量

**解决方案**：使用机器学习算法，将观察到的CEO行为的许多维度投影到两种纯粹行为类型上 (即 “领导者” 和 “管理者”)。具体来说，我们从数据集里构建起这2种纯粹行为类型，并生成单一维度的行为指数 (Behavior Index)，将每个CEO表示为两种纯粹行为的凸组合 (convex combination)

### 1\.2 Key findings 研究的主要发现

- 聘请**领导型CEO**的公司在CEO任命后会显著**提高生产率** (但随着时间的推移逐渐显现)  （原论文里 DiD model in section 4.2)
    - **领导型CEO比管理者型平均绩效更好**。CEO行为指数提升1个标准差 (更接近领导型)，则公司的销售额提升7% <-- 在控制了劳动力，资本，以及其它公司相关变量的条件下。
- 正确的**企业-CEO分配**可以提高企业绩效。 (原论文里 assignment model in section 5)
- 尽管大多数拥有管理者的公司与拥有领导者的公司一样富有成效，但总体而言，管理型CEO供过于求，因此17%的公司最终选择了“错误”类型的CEO，这种低效率的分配在低收入国家更为常见（36%对5%） (原论文里 assignment model in section 5)
- CEO与对企业的错配造成的生产率损失相当于高收入国家和低收入国家之间劳动生产率差距的13%\.  (原论文里 assignment model in section 5)
- CEO任命前的企业绩效与CEO任命后行为指标的差异不相关 (原论文里 DiD model in section 4\.2)
- 在大型企业、跨国公司、上市公司以及研发强度高、生产过程中抽象（而非常规）任务发生率较高的行业，领导者行为更为常见; 这些企业层面的差异并不能完全解释其与企业绩效的相关性。
- \.\.\.

## 2 数据和模型

> 个人认为这项研究的数据收集与建模方法非常有启发。

### 2\.1 数据

- 来自六个国家（巴西、法国、德国、印度、英国、美国）1,114位制造业CEO的高频时间记录（通过助手记录）。
- 对每个CEO记录了约50个工作小时的行为数据
- 数据包括4,253项不同的活动

因此，最终数据是一个 `1114行 * 4253列` 的矩阵，其中第 i 行第 j 列的元素是 $(i,j)$ 代表第 i 个CEO花在具有特定特征组合的活动 j 上的15分钟时间块的数量。

以下是数据集切片，其中第1行第1列的值是6，代表在跟踪其工作行为的这一周里，第1个CEO花在第1个活动上的15分钟时间块为6个。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/ceobehavior_001.PNG)

从列名可得知，这个活动即 “business\_meal\_1hrplus\_planned\_two\_plus\_board” （与董事会成员的商务聚餐，持续时间1小时以上，有提前计划，参与者数量2人以上）在这一周里一共占据了该CEO的6个15分钟时间块。因此我们可以算出总花费时间是 $15 \times 6 = 90 \text{ min}$，即一个半小时。

- 每列数据是5种特征的组合：CEO行为的“类型”(以框选的数据点为例，business\_meal)、“持续时间”(1hrplus)、“计划性”(planned)、“参与者数量”(two\_plus两人及以上)、“参与者类型”(board董事会成员)。
- 4,253项不同的活动怎么来的：就是由这5种特征组合而来

因此，这个数据集存储了每个CEO的细颗粒度行为信息。

### 2\.2 算法和模型简介：如何将4000\+列数据降维成1列

**这个研究的妙处在于**：研究人员用 LDA (Latent Dirichlet Allocation) 模型 (以及多项式混合模型) **把CEO的 4253项活动 (4253列) 降维成了 1列数据** (即CEOBehavior Index 行为指数 $\hat \theta$ ) \- 这一列行为指数 $\hat \theta$ 捕捉了每个CEO的日常活动安排的某种潜在模式。研究人员发现：

1. 当一个CEO的行为指数 $\hat \theta$ 值接近1时，这位ceo的行为更接近**领导者**型，而值更接近0时，这位CEO的行为更接近**管理者**型。
2. 并且CEO分类的确定性非常高，大部分CEO都被强烈地分类为Type1 (领导型) 或Type2 (管理型)。

随后，研究人员用这个 CEO行为指数 $\hat \theta$ 作为一个预测变量 预测了公司的生产率和利润，发现：**领导型ceo更有可能带来生产率更高、利润更高的公司**。

> "a one standard deviation in the CEO behavior index is associated with an increase of 7% in sales controlling for labor, capital, and other standard firm\-level variables"

- 翻译：**CEO行为指数提升1个标准差 (更接近领导型)，则公司的销售额提升7%**  \<\-\- 这个结论是在控制了劳动力，资本，以及其它公司相关变量的条件下获得的，因此比较扎实。



### 2\.3 模型结果探索（简版）

完整的模型探索在**3 研究代码复现**部分。这里把比较intuitive的部分抽取出来带大家先建立体感。

#### 可视化两种类别CEO的分布

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/c561fea1-2704-4b8e-a7c0-bd14888350b7.png)

**可见大部分CEO都被强烈地分类为Type1（领导型）或Type2（管理型）**，因为图 （a） 中：

- Behavior Index $\hat \theta$ 的频数最高值集中在 0 和 1 （两侧）
- 而 $\hat \theta$ 的值处在 0\.1\-0\.9 的CEO频数很少 （洼地）

这说明模型从数据中捕捉到的2种纯粹行为模式的差别很清晰。



#### 两种类别的CEO差异最大的行为

更进一步，把两类 CEO 在每一项活动上的概率做差、排序，就能看出两种行为模式各自的偏好：**Type 1 更倾向于宏观的领导与资源协调（与董事会、市场部门互动），Type 2 更倾向于管理具体的生产要素（与生产部门互动）**。

由此我们可以确认：**Type 1 是领导型，Type 2 是管理型**。完整的行为差异表见 [3.5](#35-解释模型两种ceo差别最明显的行为是)。

### 2\.4 模型技术细节

#### 第一步：Latent Dirichlet Allocation 发现CEO行为潜在模式

- 通常 LDA（隐狄利克雷分配） 都是用来做文本分析（主题模型），而如果只看数学形式，它其实可以建模任何离散数据\. 例如这个研究创新地用LDA来建模CEO行为。

以下是LDA 的最大似然估计层，即多项式因子模型（Multinomial Factor Model） 在这个案例中的数据生成机制 （Data Generating Process, DGP）: 

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/a2e6824c-d51d-49c0-b236-285edea8b30f.png)

- 也就是说，我们假设有两种纯粹的CEO类型 "Pure Behavior" （K=2）, 每个CEO都是这两种纯粹的CEO类型的混合产物。有的更偏类型1，有的更偏类型2\. 这个偏的程度由$\theta_i$决定
    - 经过对模型的interpretation, 我们认为类型1是**领导者型**，类型2是**管理者型**
- 每种纯粹CEO类型的行为特征是不一致的。在这个案例中就是 4253种行为 的概率分布向量有系统性差异。
- LDA的目的就是分离出这2种纯粹行为特征（i\.e\. 领导型 \& 管理型），并把每个CEO表示为这2种纯粹特征的按比例混合，生成其行为向量 shape `(4253, )`



**多项式因子模型（Multinomial Factor Model）**:

- 每个 CEO i 是2种潜在类别的组合 （就像“每个文档是很多主题的组合”一样）：
  $$\textbf x_i \sim Multinom(\sum_k \theta_{i,k} \mathbf{\beta_k}, C_i)$$
  -  $\theta_{i,k}$ 代表第 i 个 CEO 属于第 k 个潜在类别的权重\. $\sum_k\theta_{i,k} = 1$
  - $\mathbf{\beta_k}$代表第 k 个潜在行为特征的概率分布向量\. $\sum_j \beta_{k,j} = 1$  
    - $C_i$代表第 i 个 CEO 的行为记录总数，即 $C_i=4253$
- **在这篇论文里**： $\textbf x_i \sim Multinomial([\theta_{i,1}, \beta_1 + \theta_{i,2}, \beta_2], C_i)$
- 模型的目标是在只观察到每个CEO行为指数 $\textbf{x}_i$ 的情况下，估计 （1）每个观测实例的因子权重向量 $\theta_i = (\theta_{i,1}, \theta_{i,2}, ..., \theta_{i,K})$ 和 （2） 每个潜在因子的参数向量  $\beta_1, \beta_2, ..., \beta_K$
- 对单个 CEO $\textbf{x}_i$其概率质量函数（pmf） 即其 likelihood 为
	$$
  P(\textbf{x}_i|\theta_i, \beta) = \frac{C_i!}{x_{i1}!x_{i2}!...x_{iJ}!} \prod_{j=1}^J \left(\sum_{k=1}^K \theta_{i,k}\beta_{kj}\right)^{x_{ij}}
	$$
	其中$x_{ij}$是第 i 个实例中第 j 个特征的计数\. 在数据集中假设观测之间相互独立，总似然函数是单个似然函数的乘积：$L(\theta, \beta | \textbf{x}) = \prod_{i=1}^n P(\textbf{x}_i|\theta_i, \beta)$\. 进一步对其取对数得到其对数似然 
	$$
  \log L(\theta, \beta | \textbf{x}) = \sum_{i=1}^n \left[ \log \frac{C_i!}{x_{i1}!...x_{iJ}!} + \sum_{j=1}^J x_{ij} \log \left(\sum_{k=1}^K \theta_{i,k}\beta_{kj}\right) \right].
	$$
   在实际优化中使用EM优化算法求解 theta 和 beta。
- **最终我们将求出的**$\hat \theta$**（即CEO行为指数）放进第二步公式中作为核心predictor来预测公司的利润率**\.

#### 第二步：将CEO潜在行为模式回归到公司表现上

这一步regression包含因果推断思想，采用以下模型：
$$
y_{ifts} = \alpha \hat \theta_i + \delta^Ee_{ft}+\delta^Kk_{ft}+\delta^Mm_{ft} + \zeta_t + \eta_s + \epsilon_{ifts}
$$
- $y_{ifts}$: 企业 f 在时间 t, 国家 s 的绩效 （销售额的对数 log sales）
- **coefficient of interest**: $\alpha$
    - CEO 行为指数（$\hat{\theta}_i$）越接近“领导型”（Leader），企业可能越能够通过更高效的资源分配和战略决策来提升绩效。
- **Control Variables**:  控制潜在confounders从而希望隔离出 $\hat{\theta}_i$ 的净因果效应
    - $e_{ft}$: 企业的劳动力投入（Employment）
    - $k_{ft}$: 企业的资本投入（Capital）
    - $m_{ft}$: 企业的中间投入（Materials）
- **Fixed effect**: 控制潜在confounders从而希望隔离出 $\hat{\theta}_i$ 的净因果效应
    - $\zeta_t$: 时间固定效应（Time Fixed Effects）。控制不同年份的宏观经济环境（如经济周期、政策变化）对企业绩效的影响
    - $\eta_s$: 国家固定效应（Country Fixed Effects）。控制不同国家的制度、市场环境等系统性差异

通过加入控制变量（如 eft,kft,mft）和固定效应（$\zeta_t$,$\eta_s$），排除了其他可能影响企业绩效的因素，从而确保 $\alpha$ 反映的是 CEO 行为的“纯粹影响”。
- 因此，$\alpha$的大小应该直接表示 CEO 行为指数变化一个单位时，企业绩效（如销售额的对数）变化的幅度。

#### 细节决定成败：如何确保研究结果稳健

> 虽然不是最激动人心的部分，但却是研究成立的根基。因此有必要包括进来。

- **控制变量确保因果关系真实**：
    - 对每家公司的输入（如资本、劳动力等）和输出（如销售额等）数据进行了平均处理: 减少短期波动对分析结果的影响，并降低测量误差 （i\.e\. 更好地反映长期趋势，而不是被短期异常值干扰）
    - 使用逐年的数据（而不是三年平均数据），并将标准误差按公司进行聚类（clustering） \-\> 无论是用三年平均数据还是逐年数据，分析结果都很相似。这种一致性表明，研究结论是稳健的
    - "interview noise controls": 可能包含 "不同interviewer's scoring bias" or "CEO或其助理在回答问题时的主观性"等 \-\> 通过引入一些控制变量（如访谈员的固定效应、访谈评分的可靠性等）减少干扰
- **减轻模型依赖确保因果关系稳定**：对于Multinomial Model，不仅尝试了 Latent Dirichlet Allocation, 还用了更纯粹的 Multinomial Mixture Model （多项式混合模型）\. 还使用了K\-means, PCA等替代方法将CEOBehavior降维。发现所有模型降维出的Behavior Index与公司表现的关系上全部都是至少在0\.01的水平上显著。因此该结果没有明显的模型依赖。
- ...

## 3 研究关键部分代码复现

和论文内容保持一致，我们从数据集里去除掉出现在少于 30 位CEO工作记录的稀有行为，这样4253种特征组合（行为）降低到了654种。所以我们现在的数据集有  `1114行 * 654列`\.

**Data Generating Process**: 一共有 n=1114个CEO，每个CEO属于Type 1的可能性为 $\rho_1$ ; 属于 Type 2的可能性为 $\rho_2=(1-\rho_1)$ 。每个Type k 有 $\beta_{kj}$ 的可能性做 Activity j\. 
- $\rho_1, \rho_2$  \<\- 这两个在`multimixEM` package里被称为 "lambda"
- $\beta_1=\begin{pmatrix} \beta_{1,1} \\ \beta_{1,2} \\ ... \\ \beta_{1,654} \end{pmatrix}, \beta_2=\begin{pmatrix} \beta_{2,1} \\ \beta_{2,2} \\ ... \\ \beta_{2,654} \end{pmatrix}$ \<\- 在 `multimixEM` package 里这些被称为 "theta" 

为了解释更方便清洗，我们用稍微简单一点的 Multinomial Mixture Model （多项式混合模型） 复现该研究。它和 Multinomial Factor Model （多项式因子模型） 的区别是：

- Factor Model认为每个CEO都是一定比例“领导者”和一定比例“管理者”的混合\.
- Mixture Model则会直接给每个CEO贴上 “领导者” 或 “管理者”的标签\. All\-in\. 

因为Python里暂时没有实现MMM的现成框架，我们用R语言的 `multimixEM` package 做代码复现。

### 3\.1 Data Preprocessing

```R
library(tidyverse)
library(mixtools)
library(purrr)
library(scales) # 用于数据缩放
library(rstudioapi) # 允许 RStudio 获取当前文件路径
setwd(dirname(getActiveDocumentContext()$path))
getwd()

"
Import and clean data
"
df <- as_tibble(read.csv("../ceo_time_use.csv"))
# row: CEO_i
# col: activity_j

# ... (省略)
```

经过数据预处理，我们就把每一个列名分割成了不同特征的组合。

### 3\.2 探索性数据分析：每个特征的分布

接下来就能做一些简单的统计了，比如统计不同行为的出现次数分布、行为的持续时间的分布、行为计划 / 非计划的频数分布、行为参与者数量 & 参与者类型的频数分布等。这部分的代码省略。

因此，该数据集通过以下这些特征的组合，将CEO行为的信息有效保存了下来：

1. type 行为类型 （e\.g\. 会议, 商务聚餐, etc\.）
2. duration 行为持续时间 （15分钟, 30分钟, etc\.）
3. planning 行为是否是有计划的 （计划的 or 非计划的）; 
4. number of participants 行为的参与者数量 （1人, 多于1人）; 
5. functions of participants 行为的参与者类型。分为公司“内部人士”（如财务、市场等）和“外部人员”（如客户、银行等）。

### 3\.3 拟合模型

拟合一个多项式混合模型 `multmixEM()` 在数据集中，设置component number K=2 意味着我们假设所有行为数据的概率分布由2种不同类型CEO混合而成。

### 3\.4 解释模型：探索2种CEO类型行为特征分布的不同

> 下面部分源代码省略。但依然会用图表、文字来详尽地展示这个研究所做的数据分析。
> 

为了探索CEO行为在不同事件类型上的不同，我们把与模型里和每个特征相关的 $\beta^1_j, \beta^2_j$ 加总，创建 `theta_df`，存储 CEO 行为特征信息以及 `theta` 值 （$\beta_1, \beta_2$） ：

|                  | type 1                                  | type 2                                    |
| ---------------- | --------------------------------------- | ----------------------------------------- |
| meeting          | $\beta_2^1+\beta_3^1+...+\beta_{654}^1$ | $\beta^2_2 + \beta^2_3+...+\beta^2_{654}$ |
| business\_meal   | $\beta_1^1+...$                         | $\beta_1^2+...$                           |
| conference\_call | \.\.\.                                  | \.\.\.                                    |
| ...              |                                         |                                           |

- 每行代表一个行为，将特征（行为类别、持续时间、是否计划、参与人数、部门）抽取出来
- `type1` 和 `type2` 两列分别表示 CEO Type 1 和 Type 2 在该行为上的分布概率。

首先，我们尝试打印出模型的 `thetas` 也就是上表里的 $\beta$. `thetas` 的形状是 `[2,654]` ，说明我们做的没问题：2个纯粹类别, 每种纯粹类别里是654个行为的概率分布。

而 `thetas[k, j]` 代表 **第 k 类 CEO 在行为 j 上的期望频率** （$j \in [1,654]$）。
- 比如：下图代表Type 1 CEO在行为组合1上的期望频率为0.001；而Type 2 CEO在行为组合1上期望频率为0.0003

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/0efbd851-3e7f-4d85-8eec-fc1d4b6271c1.png)

- 模型观察 `theta[1,]` 和 `theta[2,]` 的差异，看看哪种 CEO **更倾向于某些行为**，从而归类 `Leader领导者` vs\. `Manager管理者`

于是我们创建和完善一个新的数据集 DataFrame, 包含每个CEO行为与每个部门、事件类型、持续时间等等之间关联的信息。这样，每个 CEO 行为组合都会标记与哪些部门相关联（是否与某个部门有交互）。最终得出的 theta\_df 如下：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/cb98e7a0-e1a3-4fc2-a154-22097d287ffe.png)

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/df483c0c-2973-4ff4-83c4-b2760adc6e88.png)

接下来我们可以**可视化 CEO 行为类别的分布**。首先是活动持续时间的分布：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/eda1a4c7-077f-4bfc-953a-dda8db9b438e.png)

- 观察结果：Type 2 CEO 参加的活动持续时间更长。但没有很明显的区别



然后是两种CEO计划性的分布：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/1a9530ca-a3f7-45a1-a625-f3dcbd79b460.png)

- Type 1 CEO 更倾向于有计划的活动，这个区别很明显



活动参与人数的分布：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/59b863a7-60af-4c6b-b740-0b58eaba2a28.png)

- Type 1 CEO 更倾向于一对一会议; Type 2 CEO 更倾向于大规模会议



部门分布：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/8a5a0dab-fd62-4e6b-b2d2-d5ed09ba8307.png)

- Type 1 CEO 更常与其他高管（board, admin）互动；Type 2 CEO 更常与运营\&生产部门（production）互动



### 3\.5 解释模型：两种CEO差别最明显的行为是？

下表为两种类别的CEO差异最大的行为列表。我们观察：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/a8ef1797-1c5d-4145-8dbb-1954232866fb.png)

- Type 2 CEO 做的更多的事情（从上往下看）：
    1. meeting\.1hrplus\.planned\.two\.plus\.production \-0\.033769504
    2. meeting\.1hrplus\.planned\.two\.plus\.mkting\.production \-0\.015153549
    3. meeting\.1hrplus\.planned\.one\.clients \-0\.013418091

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/a35c7292-1cbd-4241-b098-229c0d335aec.png)

- Type 1 CEO 做的更多的事情 （从下向上看）
    1. meeting\.1hrplus\.planned\.two\.plus\.mkting 0\.031288705
    2. meeting\.1hrplus\.planned\.two\.plus\.mkting\.clients 0\.011079017
    3. meeting\.1hrplus\.planned\.two\.plus\.board 0\.010672928

所以Type 1更倾向于宏观的领导\&资源协调。Type 2更倾向于管理具体生产要素。

- 因此，我们可以得出：**Type 1 是领导型，Type 2 是管理型**


### 3\.6 解释模型：领导型/管理型 各占比多少？

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/746304c9-28d3-4e64-bf3b-61db639cc370.png)

- 可见领导型只占35%\. **领导型人才更稀缺**



### 3\.7 解释模型：每位CEO在多大程度上属于领导者/管理者？

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/50174ee2-5d7c-4881-834b-d1f06d7efdf8.png)

- 其中，`prob1`该 CEO 属于 Type 1 的概率，`prob2`是该 CEO 属于 Type 2 的概率
- 可以见到总共1114行，对应1114个CEO，数据是没有问题的。

然后看看每个Type的内部的概率密度分布曲线。**观察要点**：如果大多数 `prob1` 接近 1，说明 Type 1 CEO **被强烈归类为这一类别**；如果较分散，说明分类不确定性较高。`prob2` 同理

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/ee6db264-9d38-4ea4-9b23-64c23818ec3f.png)

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/9a3d2a6b-e769-4777-9788-2663c77376bc.png)

- 可以看到，分类的确定性相当高。几乎全部的Type1 CEO都被强烈归类为Type 1\. Type 2也一样。



## 4 一点个人想法

**作为降维方法，LDA对比PCA、K\-means在某些场景有显著优势**。LDA（以及MMM, MFM）的优势在于其保留双层的信息：表层是CEO行为指数（以及它的不同值所对应的CEO类型），底层是每一个类型内的行为概率分布。所以我们能做到「解释」每一个指数是什么。但单纯的PCA \& K\-means的分类标准难以解释，不知道实际上降了个什么维。



## 5 附录：MMM, MFM \& LDA的关系

**Multinomial Mixture Model 是什么**:

- 共 N 个CEO，每一个CEO的行为是一个向量 $\textbf{x}_i$\. 对于每一个CEO i, we have a latent cluster $z_i \in \{1, 2, ..., K\}$ drawn from probability vector $\rho \in \mathbb R^K$ and $\sum^K_k \rho_k=1$
- 因此$\textbf x_i|z_i = k \sim Multinom(\beta^k, N)$ : 给定潜在类别 $z_i = k$，观测值 $\textbf{x}_i$ 是从对应的多项式分布 $Multinom(\beta^k, N)$ 中抽取的\.
- 这个案例里，k 是 "Leader" OR "Manager"\. 
- MMM的目标是在只观察到 $\textbf{x}_i$ 的情况下，估计参数$(\rho, \beta^1, \beta^2, ..., \beta^K)$，这通常通过EM算法（Expectation\-Maximization）来实现。
    - $\rho$: vector of K dimensions （在文本分析中是K latent topics；在本案例中是K=2，两种CEO类型）
    - $\beta$: matrix of K 乘 J shape （K categories 乘 J possible outcomes in each category）
- 似然函数：单个 文档内词频分布/CEO行为的概率分布$\textbf x_i$的似然为 $\sum_{k=1}^K P(z_i=k) \cdot P(\textbf x_i|z_i=k)$ \. 所有观测独立，所以整体似然为单个似然的连乘 
	$$
    \prod (\sum_{k=1}^K P(z_i=k) \cdot P(\textbf x_i|z_i=k))
    $$
    将其展开则得到MMM的 Likelihood function: 
    $$
    L(\textbf x | \rho, \beta) = \prod_{i=1}^n\left( \sum_{k=1}^K\rho_k \cdot \frac{N_i!}{x_{i1}!,...x_{iJ}!} \prod_{j=1}^J (\beta_j^{k})^{x_{ij}} \right)
    $$
    其中：
    - $P(z_i=k)$ 代表: $\sum_{k=1}^K\rho_k$ ，这是所有类别的先验概率之和 （$\rho_k$是每个component k 的先验概率）
    - $P(\textbf x_i|z_i=k)$ 代表: $(\beta_j^{k})^{x_{ij}}$ ，这是在给定第 k 个类别下，观测到当前样本 $\textbf x_i$ 的概率
- 对Likelihood function使用EM算法，可以实现对参数$(\rho, \beta^1, \beta^2, ..., \beta^K)$的有效估计。

**从 Multinomial Mixture Model 到 Multinomial Factor Model**: 

- Mixture: $\textbf x_i \sim Multinom(\beta^k, N)$ **目标**：将数据分割成不同的子总体或类别
- Factor: $\textbf x_i \sim Multinom(\sum_k \theta_{i,k} \mathbf{\beta_k}, C_i)$ **目标**：发现数据中的潜在维度或特征

**从Multinomial Factor Model 到 Latent Dirichlet Allocation**: 对于MFM我们有 $\textbf x_i \sim Multinom(\sum_k \theta_{i,k} \mathbf{\beta_k}, C_i)$

- Blei et al\., （2003） 提出 LDA，通过给多项式概率向量添加 Dirichlet 先验分布，进一步扩展多项式因子模型：
    - $\theta_i \sim Dirichlet(\alpha)$
    - $\beta_k \sim Dirichlet(\eta)$
- 好处：将MFM扩充成完整贝叶斯模型而不仅是MLE \-\-\> 我们可以获得参数的后验分布，而不仅仅是点估计; 防止overfit; 平滑估计；
    - 以及可控的主题分布：通过调整Dirichlet参数α，我们可以控制文档中主题分布的集中度或分散度。低α值倾向于稀疏的主题分布（文档倾向于集中在少数几个主题上），高α值则倾向于均匀分布, etc\.

## 6 结论速览

本文充斥着比较多的数学推导，非专业读者可以快速掠过。因此在这里，将本文中出现的对非专业读者而言（可能）比较有意思的研究结论做一个集合：

1. **领导型CEO带领的公司绩效更好**。平均而言，CEO行为指数提升1个标准差 (更接近领导型)，则公司的销售额提升**7%**. 这个结论是在控制了劳动力，资本，以及其它公司相关变量的条件下，因此具备统计学显著性、因果价值。
2. 聘请领导型CEO的公司在CEO任命后会显著提高生产率 (但随着时间的推移逐渐显现) 
3. 领导型CEO只占全部样本的的35%\. **领导型人才更稀缺**
4. 总体而言，管理型CEO供过于求，因此17%的公司最终选择了“错误”类型的CEO，这种低效率的分配在低收入国家更为常见（36%对5%） 
5. 在大型企业、跨国公司、上市公司以及研发强度高、生产过程中抽象（而非常规）任务发生率较高的行业，领导者行为更为常见; 这些企业层面的差异并不能完全解释其与企业绩效的相关性。
6. **领导型 CEO 的行为特征**：更倾向于有计划的活动、更倾向于一对一会议而非大规模会议。领导型更常与其他高管（board, admin）互动，而管理型 CEO 更常与运营&生产部门 (production) 互动。 【注意：还有更多的区别，在正文里。这里不一一列举了】
7. .\.\.
