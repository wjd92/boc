
title: DBSCAN算法原理

date: 2018-09-25 16:00:00

tags: 
  - 聚类

categories:
  - 机器学习

mathjax: true

top: 2

---
<img src="https://images.bumpchicken.cn/img/20220508173927.png" width="60%" height="20%">

DBSCAN(Density-Based Spatial Clustering of Application with Noise)是一种基于密度的空间聚类算法。该算法将具有足够密度的区域划分为簇，并在具有噪声的空间数据中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。算法无需事先指定聚类中心数目，可以对大规模无规则形状的数据进行有效聚类。

<!--more-->

## 相关定义
DBSCAN有自己的一套符号体系，定义了许多新概念，数学体系十分严谨

### 密度定义

给定数据集$D$

- $\epsilon$: 邻域半径
- $\epsilon$-邻域: 邻域内点的集合

$$N_{\varepsilon}(p):=\{\text{q in dataset D} \mid \operatorname{dist}(p,q)<=\varepsilon\}$$

【注】_距离度量$dist(p,q)$是聚类算法中一个值得探究的问题。此处的距离度量可以为欧氏距离、曼哈顿距离等多种距离度量方式，并且数据点的维度可为任意维度_

- MinPts: 核心点邻域内数据点的最小数量

<img src="https://images.bumpchicken.cn/img/20220508215638.png" width="50%" height="80%">

如上图，当MinPts = 4, p的密度相较于q大，p称为高密度点

### 核心点、边界点和离群点定义

<img src="https://images.bumpchicken.cn/img/20220508220041.png" width="60%" height="60%">

- 核心点(Core): 高密度点，其 $\epsilon$-邻域数据点数量 >= MinPts
- 边界点(Border): 低密度点，但在某个核心点的邻域内
- 离群点(Outlier): 既不是核心点也不是边界点

### 密度可达定义

- 直接密度可达： 如果p是一个核心点，切q在p的$\epsilon$-邻域内，那么称q直接密度可达p

【注】_不能说p直接密度可达q，直接密度可达不具有对称性(symmetric)_

<img src="https://images.bumpchicken.cn/img/20220508221408.png" width="40%" height="40%">


- 密度可达:如果存在一串这样的数据点: $p_{1},p_{2},...,p_{n}$，其中$p_{1}=q,p_{n}=p$,且$p_{i+1}$直接密度可达$p_{i}$，那么称p密度可达q

【注】_不能说q密度可达p，密度可达同样不具有对称性_

<img src="https://images.bumpchicken.cn/img/20220508222514.png" width="50%" height="50%">

### 密度连通

如果p和q都密度可达点o，那么称p和q密度连通，如下所示

<img src="https://images.bumpchicken.cn/img/20220508224900.png" width="50%" height="50%">

【注】_密度连通具有对称性，可以说q和p密度连通_

## 聚类准则
给定一个数据集D，参数$\epsilon$和MinPts，那么聚类产生的子集C必须满足两个准则：

1. Maximality(极大性)：对于任意的p、q，如果$p\in C$，且q密度可达p，那么同样$q\in C$
2. Connectivity(连通性)：对于任意的p、q，p和q是密度相连的

## 聚类流程

DBSCAN聚类过程如下图所示

<img src="https://images.bumpchicken.cn/img/20220508225615.png" width="80%" height="80%">


## 参数选择

### 邻域大小$\epsilon$

DBSCAN采用全局$\epsilon$和MinPts值，因此每个节点的邻域大小是一致的。当数据密度和聚簇间距离分布不均匀时，若选取较小的$\epsilon$，则较稀疏的聚簇中的数据点密度会小于MintPts，而被认为是边界点而不被用于所在类的进一步扩展。可能导致较稀疏的聚簇被划分为多个性质相似的小聚簇。相反,若选取较大的$\epsilon$，则离得较近而密度较大的那些聚簇可能被合并为同一个聚簇，他们之间的差异将被忽略。因此这种情况下，选取合适的邻域大小是较为困难的，当维度较高时，$\epsilon$的选取更加困难

### MinPts

参数MinPts的选取有一个指导性原则，即 $MinPts >= dim+1$，这里$dim$表示聚类空间的维度大小

## 优缺点

### 优点
1. 可以对任意形状的稠密数据集进行聚类（K-Means一般只适用于凸数据集）
2. 可以在聚类时发现异常点，对数据集的异常点不敏感

### 缺点
1. 如果样本集的密度不均匀，聚类间距相距很大时，聚类效果较差
2. 对于参数 $\epsilon$ 和MinPts敏感，不同参数组合对聚类效果影响较大