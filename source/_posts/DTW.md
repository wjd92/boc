
title: 动态归整距离（DTW）详解

date: 2019-04-08 19:51:00

tags: 
  - 时间序列分析

categories:
  - 时间序列分析

mathjax: true

---

DTW(Dynamic Time Warping)动态归整距离是一种衡量两个时间序列相似度的方法

<!--more-->

## DTW原理

在时间序列分析中，需要比较相似性的两段时间序列的长度可能并不相等，在语音识别领域表现为不同人的语速不同。而且同一个单词内的不同音素的发音速度也不同，比如有的人会把‘A’这个音拖得很长，或者把‘i’发的很短。另外，不同时间序列可能仅仅存在时间轴上的位移，亦即在还原位移的情况下，两个时间序列是一致的。在这些复杂情况下，使用传统的欧几里得距离无法有效地求的两个时间序列之间的距离(或者相似性)。

__DTW通过把时间序列进行延伸和缩短，来计算两个时间序列性之间的相似性__

<img src="https://images.bumpchicken.cn/img/20220509012950.png" width="50%" height="50%">

如上图所示，上下两条实线代表两个时间序列，它们之间的虚线代表两个时间序列之间相似的点。DTW使用所有这些相似点之间距离的和，称之为归整路径距离（Warp Path Distance)

## DTW计算方法

令要计算相似度的两个时间序列分别为$X$和$Y$，长度分别为$|X|$和$|Y|$

### 归整路径(Warp Path)

归整路径的形式为$W=w_{1},w_{2},...,w_{k}$,其中$Max(|x|,|Y|)<= K <= |X| + |Y|$
$w_{k}$的形式为$(i,j)$，其中$i$表示的是$X$中的$i$坐标，$j$表示的是$Y$中的$j$坐标。
归整路径$W$必须从$w_{1}=(1,1)$ 开始，到$w_{k}=(|X|,|Y|)$结尾，以保证$X$和$Y$中的每个坐标都在$W$中出现。
另外，$w_{k}$中$(i,j)$必须是单调增加的，以保证上图中的虚线不会相交，所谓单调增加是指：

$$w_{k}=(i,j), w_{k+1}=(i^{'},j^{'}) \qquad i<=i^{'}<=i+1, j<=j^{'}<=j+1$$

最后得到的归整路径是距离最短的一个路径:

$$Dist(W)=\sum^{k=K}_{k=1}Dist(w_{ki}, w_{kj})$$

其中$Dist(w_{ki}, w_{kj}$为任意经典的距离计算方法，比如欧氏距离。$w_{ki}$是指$X$的第i个数据点，$w_{kj}$是指$Y$的第$j$个数据点

### DTW实现

在实现DTW时，我们采用动态规划的思想，令$D(i,j)$表示长度为$i$和$j$的两个时间序列之间的归整路径距离:

$$D(i,j)=Dist(i,j)+min[D(i-1,j),D(i,j-1),D(i-1,j-1)]$$

<img src="https://images.bumpchicken.cn/img/20220509015951.png" width="80%" height="30%">

代码如下:
```python
import sys

def distance(x,y):
   """定义你的距离函数，欧式距离，街区距离等等"""
   return abs(x-y)
  
def dtw(X,Y):
    M=[[distance(X[i],Y[j]) for i in range(len(X))] for j in range(len(Y))]
    l1=len(X)
    l2=len(Y) 
    D=[[0 for i in range(l1+1)] for i in range(l2+1)]
    D[0][0]=0 
    for i in range(1,l1+1):
      D[i][0]=sys.maxint
    for j in range(1,l2+1):
      D[0][j]=sys.maxint
    for i in range(1,l1+1):
      for j in range(1,l2+1):
        D[i][j]=M[i-1][j-1]+min(D[i-1][j],D[i][j-1],D[i-1][j-1])
    return D[l1][l2]
```

__DTW采用动态规划实现，时间复杂度为$O(N^{2})$,有一些改进的快速DTW算法，如FastDTW[1], SparseDTW, LB_Keogh, LB_Imporved等__

## 参考资料

1. FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador, Philip Chan.