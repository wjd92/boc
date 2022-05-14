
title: MicroCause论文阅读笔记

date: 2020-09-07 17:30:00

tags: 
  - 根因定位

categories:
  - 论文阅读笔记

mathjax: true

---

该论文是清华大学NetMan实验室2020年发表与IWQoS上的一篇关于微服务根因定位的论文。论文题目是《Localizing Failure Root Causes in a Microservice through Causality Inference》/ 通过根因推理对微服务进行根因定位。论文提出一种基于改进的PC算法和随机游走的根因定位算法。

<!--more-->

## 问题定义

### 典型的微服务架构

<img src="https://images.bumpchicken.cn/img/20220515004031.png" width="60%">

如上是一个折扣券服务，该服务会有上游和下游的依赖中间件、服务等。运维人员会对每个微服务的状态进行监控，生成一些指标，如响应时间（RT），QPS，CPU利用率等。当某个故障发生时，体现在这些指标上就是一系列异常抖动

<img src="https://images.bumpchicken.cn/img/20220515004250.png" width="60%">

如上是一个故障，W-QPS指标导致了这次case，为了找到如W-QPS这种根因指标，论文提出了一种根因定位方法：MicroCause

### 面临的挑战与解决思路

通常解决上述根因定位问题基本思路是：首先构建各个组件之间的依赖关系图然后通过随机游走（random walk）的方式寻找根因节点。此方法面临两个主要问题：

1. __传统因果图构造方法适用于独立且均匀分布的数据，无法充分利用传播延迟__
2. __当前随机游走方法基于这样的假设：与KPI异常相关性更高的指标更有可能是根因指标，但是实际场景中并不总是如此__

为了解决上述问题，论文提供了以下解决方法：

- 设计一种新的因果图构造方法：__PCTS（path condition time series)__。首先通过改进的PC算法对时序数据学习得到一个因果图，然后对节点进行边(edge)的生成

- 设计一种新的随机游走方法：__TCORW（temporal cause oriented random walk）__。在TCORW中，主要利用了三类信息：

  1. 指标间的因果关系

  2. 指标当前是否异常以及异常程度
  3. 指标的优先级（通过先验经验确定）

## 因果图生成算法与随机游走算法

### PC算法

由Peter Spirtes 和Clark Glymour提出的**PC算法**是当前比较广泛使用的构造因果图的算法，PC算法旨在了解随机变量之间的因果关系。假设我们准备学习M个随机变量的因果图，输入是N个独立的均匀分布的样本，每个样本包含M个值，分别代表M个随机变量的观测值。PC算法将输出具有M个节点的有向无环图（DAG），其中每个节点代表一个随机变量（每个时间序列视为一个随机变量，并将每个时间点的数据视为样本）。PC算法基于以下假设：变量之间没有边（相互独立）。给定变量集S，A独立于B，表示为A⊥B，PC算法包含下列步骤：

1. 构建一个M个随机变量的完全连接图。
2. 对每个相邻变量在显著性水平α下进行条件独立性测试。如果存在条件独立性，则删除两个变量之间的边。在这个步骤中，条件变量集S的大小逐步增加，直到没有更多的变量可以添加到S为止。
3. 根据v-structure确定某些边的方向
4. 确定其余边的方向

## 随机游走算法

随机游走算法可以在各种类型的空间（图形、向量等）执行随机游走，通常包含下列步骤：

1. 生成关系图G，其中V是节点集，E是边集

2. 计算矩阵Q

   a)前向游走（从结果节点到根因节点）：特别地，我们假定异常节点更可能是根因节点，因此

      $$Q_{ij} = R(V_{abnormal}, v_{j})$$

      其中$R(v_{abnormal}, v{j})$是$v_{abnormal}$和$v_{j}$的相关系数

   b) 后向游走（从根因节点到结果节点）：此步是为了避免算法陷入与异常节点相关性较低的节点

    $$Q_{ji} = \rho R(v_{abnormal, v_{i}})$$

    其中$\rho \in [0, 1]$

   c) 原地游走（从源节点到源节点）：如果算法走到的节点都与异常节点相关性较低，则该点可能表示根本原因

   $$Q_{ii} = max[0, R(v_{abnormal}, v_{i}) - \max _{k: e_{k i} \in E} R\left(v_{a b n o r m a l}, v_{k}\right)]$$

3. 归一化Q的每一行

   $$\bar{Q}_{i j}=\frac{Q_{i j}}{\sum_{j} Q_{i j}}$$

4. 在G上进行随机游走，从$v_{i}$到$v_{j}$的概率是$Q_{ij}$

通过以上四个步骤，被访问最频繁的节点就是根因节点

##  MicroCause

论文提出了一种MicroCause算法，下图是其整体结构：

<img src="https://images.bumpchicken.cn/img/20220515004449.png">

 当某个KPI检测到异常时，MicroCause会被启动进行根因分析。所有相关指标最近4h的数据将作为MicroCause的输入，通过PCTS生成因果图，以及检测相关指标是否异常和异常程度，然后通过TCORW进行根因分析，给出TOP N的根因节点

### PCTS

对于一个故障X，给定一个数据集$\mathbf{I}_{t}^{i}, t=0, \ldots, T, i=1, \ldots, N$即N个时间序列，每个时间序列长度为T。定义最大的时间延迟$\tau_{\max }$即如果我们想找到$I_{t}^{i}$的根因时，$I_{t}$到$I_{t-\tau _{max}}$的时间序列会被使用到，利用滑动窗口对样本进行独立性测试：

1. 提取每个时间点最大延时时间内的数据, 生成一个父集合$\widehat{\mathcal{P}}\left(I_{t}^{i}\right)=\left(\mathbf{I}_{t-1}, \ldots, \mathbf{I}_{t-\tau_{\max }}\right)$
2. 和PC算法类似，对$\widehat{\mathcal{P}}\left(I_{t}^{i}\right)$中每个时间点进行独立性测试，如果某个点不满足置信度$\alpha_{I P C}$，要求，则将其从$\widehat{\mathcal{P}}\left(I_{t}^{i}\right)$中移除
3. 构建出以下结构的图G，每个指标每个时间点都是一个节点，下面的图还不能用来定位，需要将其转化为以指标为节点因果图的DAG（Fig 2），但是由于有了延迟传播的概念，因果图将更加符合实际情况。

<img src="https://images.bumpchicken.cn/img/20220515004554.png" width="60%">

## TCORW

包含三个步骤：

**Step1: 面向根因的随机游走**

传统随机游走过程中，通常使用相关性来量化指标与异常指标的关系，相关研究表明：相关并不等于因果关系。因为相关关系无法消除第三个变量的影响（在因果关系研究中称为混杂因素）。 在面向原因的随机游动中，论文通过偏相关来计算矩阵Q，它可以消除混杂因素的影响。我们在随机游动中计算矩阵Q如下：

1）前向游走（从结果节点到根因节点）：

$$Q_{i j}=R_{p c}\left(v_{a k}, v_{j} \mid P a\left(v_{a k}\right) \backslash v_{j}, P a\left(v_{j}\right)\right)$$

其中，$R_{pc}$代表偏相关，使用了和皮尔逊相关系数相关的算法。$P_{a}(v_{ak})$是$v_{ak}$的父节点集，$P a\left(v_{a k}\right) \backslash v_{j}$表示$v_{j}$将从$v_{ak}$的父节点集中删除。我们将$P a\left(v_{a k}\right) \backslash v_{j}$和$Pa（v_{j}）$作为偏相关的混杂因素

2）后向游走（从根因节点到结果节点）

$$Q_{j i}=\rho R_{p c}\left(v_{a k}, v_{i} \mid P a\left(v_{a k}\right) \backslash v_{i}, P a\left(v_{i}\right)\right)$$

其中，$\rho$是用来控制后向游走的权重系数，$\rho \in [0, 1]$

3）原地游走

$$\begin{array}{l}
Q_{i i}=\max \left[0, R_{p c}\left(v_{a k}, v_{i} \mid P a\left(v_{a k}\right) \backslash v_{i}, P a\left(v_{i}\right)\right)-P_{p c}^{m a x}\right] \\
P_{p c}^{\max }=\max _{k: e_{k i} \in E} R_{p c}\left(v_{a k}, v_{k} \mid P a\left(v_{a k}\right) \backslash v_{k}, P a\left(v_{k}\right)\right)
\end{array}$$

同样地，归一化矩阵Q：

$$\bar{Q}_{i j}=\frac{Q_{i j}}{\sum_{j} Q_{i j}}$$

**Step 2  潜在根因节点打分**

定义指标的根因节点得分公式如下：

$$\gamma_{i}=\lambda \bar{c}_{i}+(1-\lambda) \bar{\eta}_{\max }^{i}$$

其中，$\bar{c}_{i}$是在随机游走中被访问的次数（标准化过的）,$\bar{\eta}_{\max }^{i}$是指标的异常程度（经标准化过的），$\lambda$是用来控制两者权重的系数，$\lambda \in [0,1]$

**Step 3 根因节点排序**

把指标分成了三类，如下表所示，通常高级别的指标会影响低级别的指标，因此，在确定根因节点时高级别指标会拥有更高的权重系数。

<img src="https://images.bumpchicken.cn/img/20220515004925.png">

论文考虑了指标的优先级、指标的异常时间，指标的异常根因得分，提出了一个根因节点排序算法，如下：

<img src="https://images.bumpchicken.cn/img/20220515005008.png" width="60%">

即级别更高的指标、更加异常的指标，更早发生异常的指标更有可能会被认为是根因节点

## 算法结果验证

与相关算法对比

<img src="https://images.bumpchicken.cn/img/20220515005103.png" width="80%">

## 参考资料

1.Meng Y, Zhang S, Sun Y, et al. Localizing failure root causes in a microservice through causality inference[C]//2020 IEEE/ACM 28th International Symposium on Quality of Service (IWQoS). IEEE, 2020: 1-10.

