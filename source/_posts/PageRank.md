
title: PageRank算法原理

date: 2020-12-08 18:05:00

tags: 
  - 排序

categories:
  - 根因定位

mathjax: true

------
PageRank, 网页排名，又称网页级别，佩奇排名等，是一种由搜索引擎根据网页之间相互的超链接计算的技术，作为网页排名的要素之一，以Google创办人拉里佩奇(Larry Page)命名。Google用它来体现网页的相关性和重要性，在搜索引擎优化操作中是经常被用来评估网页优化的成效因素之一。

<!--more-->

## PageRank简单形式

### 基本思想

- 如果一个网页被其他很多网页链接，则重要，权值高
- 如果PageRank值高的网页链接某个网页，则该网页权值也会相应提高

### 计算方式

假设由如下四个网页A、B、C、D，链接信息如下图所示:

<img src="https://images.bumpchicken.cn/img/20220508234438.png" width="50%" height="50%">

上图是一个有向图，将网页看成节点，网页之间的链接关系用边表示，出链指的是链接出去的链接，入链指的是进来的链接，比如上图A有2个入链，3个出链

__PageRank定义__: 一个网页的影响力 = 所有入链集合的页面加权影响力之和

上图A节点的影响力可用如下公式计算:

$$PR(A) = \frac{PR(B)}{L(B)} + \frac{PR(C)}{L(C)} + \frac{PR(D)}{L(D)}$$

其中，$PR(A)$表示网页A的影响力，$L(B)$表示B的出链数量，用通用的公式表示为：

$$PR(u)=\sum_{\nu \in B_{u}} \frac{P R(v)}{L(v)}$$

u为待评估的页面，$B_{u}$为页面u的入链集合。针对入链集合中的任意页面v，它能给u带来的影响力是其自身的影响力$PR(v)$除以v页面的出链数量，即页面v把影响力$PR(v$平均分配给了它的出链，这样统计所有能给u带来链接的页面v，得到的总和就是网页u的影响力，即为$PR(u)$

因此，PageRank的简单形式定义如下：

> 当含有若干个节点的有向图是强连通且非周期性的有向图时，在其基础上定义的随机游走模型，即一阶马尔科夫链具有平稳分布，平稳分布向量称为这个有向图的PageRank。若矩阵M是马尔科夫链的转移矩阵，则向量R满足:
> $$ MR = R $$

上图A、B、C、D四个网页的转移矩阵M如下:

$$M=\left[\begin{array}{cccc}
0 & 1 / 2 & 1 & 0 \\
1 / 3 & 0 & 0 & 1 / 2 \\
1 / 3 & 0 & 0 & 1 / 2 \\
1 / 3 & 1 / 2 & 0 & 0
\end{array}\right]$$

假设A、B、C、D四个页面的初始影响力是相同的，即$w_{0}^{T} = [1/4\space1/4\space1/4\space1/4]$

第一次转移后，各页面影响力$w_{1}$变为:

$$w_{1}=M w_{0}=\left[\begin{array}{cccc}
0 & 1 / 2 & 1 & 0 \\
1 / 3 & 0 & 0 & 1 / 2 \\
1 / 3 & 0 & 0 & 1 / 2 \\
1 / 3 & 1 / 2 & 0 & 0
\end{array}\right]\left[\begin{array}{c}
1 / 4 \\
1 / 4 \\
1 / 4 \\
1 / 4
\end{array}\right]=\left[\begin{array}{l}
9 / 24 \\
5 / 24 \\
5 / 24 \\
5 / 24
\end{array}\right]$$

之后再用转移矩阵乘以$w_{1}$得到$w_{2}$，直到第n次迭代后$w_{n}$收敛不再变化，上述例子，$w$会收敛至[0.3333 0.2222 0.2222 0.2222]，对应A、B、C、D的影响力

### 等级泄露和等级沉没

1. 等级泄露（Rank Leak): 如果一个网页没有出链，就像是一个黑洞一样，吸收了其他网页的影响力而不释放，最终会导致其他网页的PR值为0，如下图所示:

<img src="https://images.bumpchicken.cn/img/20220509000856.png" width="50%" height="50%">

2. 等级沉没（Rank Sink): 如果一个网页只有出链没有入链，计算过程迭代下来，会导致这个网页的PR值为0，入下图所示:

<img src="https://images.bumpchicken.cn/img/20220509001111.png" width="50%" height="50%">

## PageRank改进版

为了解决简化模型中存在的等级泄露和等级沉没问题，拉里佩奇提出了PageRank的随机浏览模型。他假设了这样一个场景：

> 用户并不都是按照跳转链接的方式来上网，还有一种可能是不论当前处于哪个页面，都有概率访问到其他任意页面，比如用户就是要直接输入网址访问其他页面，虽然这个概率比较小

所以他定义了阻尼因子d，这个因子代表了用户按照跳转链接来上网的概率，通常可以取一个固定值0.85，而$1-d=0.15$则代表了用户不是通过跳转链接的方式来访问网页的概率

下式是PageRank计算影响力的改进公式:

$$PR(u)=\frac{1-d}{N}+d \sum_{\nu=B_{u}} \frac{P R(v)}{L(v)}$$

其中，N为网页总数，这样我们有可以重新迭代网页的权重计算了，因为加入了阻尼因子d，一定程度上解决了等级泄露和等级沉没的问题

同样地，定义概率转移矩阵M，则其一般公式如下:

$$R=d M R+\frac{1-d}{n}1$$

其中，$d(0<=d<=1)$为阻尼因子，$1$是所有分量为1的n维向量

## PageRank 代码实现

```python
import numpy as np
from scipy.sparse import csc_matrix

def pageRank(G, s=.85, maxerr=.0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    n = G.shape[0]
    # transform G into markov matrix A
    A = csc_matrix(G, dtype=np.float)
    rsums = np.array(A.sum(1))[:, 0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums == 0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r - ro)) > maxerr:       # 迭代直至收敛
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0, n):
            # inlinks of state i
            Ai = np.array(A[:, i].todense())[:, 0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)
            r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))

    # return normalized pagerank
    return r / float(sum(r))
```

使用示例：
```python
G = np.array([[0,0,1,0,0,0,0],
              [0,1,1,0,0,0,0],
              [1,0,1,1,0,0,0],
              [0,0,0,1,1,0,0],
              [0,0,0,0,0,0,1],
              [0,0,0,0,0,1,1],
              [0,0,0,1,1,0,1]])
print(pageRank(G,s=.86))
--------------------
[0.12727557 0.03616954 0.12221594 0.22608452 0.28934412 0.03616954 0.16274076]
```

## 参考资料

1.https://www.cnblogs.com/jpcflyer/p/11180263.html

2.[PageRank notebook](https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC21%E7%AB%A0%20PageRank%E7%AE%97%E6%B3%95/21.PageRank.ipynb)

## 其他

一些相似算法或改进的算法

1.LeaderRank

2.Hilltop算法

3.ExpertRank

4.HITS

5.TrustRank
