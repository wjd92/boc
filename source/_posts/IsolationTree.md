
title: 孤立森林原理详解

date: 2020-05-25 11:33:00

tags: 
  - 异常检测

categories:
  - 异常检测

mathjax: true

---
  
![](https://images.bumpchicken.cn/img/tree.png)

孤立森林（Isolation Forest）是周志华团队于2008年提出的一种具有线性复杂度的异常检测算法，被工业界广泛应用于诸如异常流量检测,金融欺诈行为检测等场景。

<!--more-->

## 算法原理
异常检测领域，通常是正常的样本占大多数，离群点占绝少数，因此大多数异常检测算法的基本思想都是对正常点构建模型，然后根据规则识别出不属于正常点模型的离群点，比较典型的算法有One Class SVM(OCSVM), Local Outlier Factor(LOF)。和多数异常检测算法不同，孤立森林采用了一种较为高效的异常发现算法，其思路很朴素，但也足够直观有效。

考虑以下场景，一个二维平面上零零散散分布着一些点，随机使用分割线对其进行分割，直至所有但点都不可再划分（即被孤立了）。直观上来讲，可以发现那些密度很高的簇需要被切割很多次才会停止切割，但是密度很低的点很快就会停止切割到某个子空间了。

![](https://images.bumpchicken.cn/img/20220424235501.png)

图1  孤立森林原理示意图

## 训练

### 构建iTree

### 构建IForest

## 评估

## 参考资料

1.Liu F T, Ting K M, Zhou Z H. Isolation forest[C]//2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008: 413-422.
