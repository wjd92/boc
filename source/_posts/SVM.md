
title: 深入浅出SVM

date: 2019-09-26 00:09:00

tags: 
  - 分类

categories:
  - 机器学习

mathjax: true

top: 8

---
<img src="https://images.bumpchicken.cn/img/20220511220448.png" width="60%">
Support Vector Machine（支持向量机），是一种非常经典的机器学习分类方法。
<!--more-->

## SVM基本原理

Support Vector Machine（支持向量机），是一种非常经典的机器学习分类方法。它有严格的数学理论支持，可解释性强，不依靠统计方法，并且利用核函数技巧能有效解决一些线性不可分的场景。

给定样本集：$D=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right),\left(\boldsymbol{x}_{2}, y_{2}\right), \ldots,\left(\boldsymbol{x}_{m}, y_{m}\right)\right\}, y_{i} \in\{-1,+1\}$，如下所示，假设左上方的点为正样本，右下方的点为负样本

<img src="https://images.bumpchicken.cn/img/20220511220958.png" width="60%">


 寻找一个最优的分类面，使得依赖这个分类面产生的分类结果最具鲁棒性，体现在图上就是分类样本离这个分类平面尽可能远，具有最大的间隔。将这个分类的平面称为**最大间隔超平面**，离这个最大间隔超平面最近的点称之为**支持向量（Support Vector)，分类超平面的构建只与这些少数的点有关，这也是为什么它叫作支持向量机的原因。**

<img src="https://images.bumpchicken.cn/img/20220511221109.png" width="60%">


## SVM最优化问题

SVM 目的是找到各类样本点到超平面的距离最远，也就是找到最大间隔超平面。任意超平面可以用下面这个线性方程来描述：

$$\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b=0$$

我们知道，二维空间点(x, y) 到直线 Ax+By+C=0的距离公式为：

$$\frac{|A x+B y+C|}{\sqrt{A^{2}+B^{2}}}$$

扩展到n维空间，点$x=\left(x_{1}, x_{2} \dots x_{n}\right)$到$w^{T} x+b=0$距离为：

$$\frac{\left|w^{T} x+b\right|}{\|w\|}$$

其中，$\|w\|=\sqrt{w_{1}^{2}+\ldots w_{n}^{2}}$

假定分割超平面能将样本点 __准确__ 分为两类，设支持向量到最大间隔超平面的距离为d，则其他向量到最大间隔超平面的距离大于d。于是有:

$$\left\{\begin{array}{l}{\frac{w^{T} x+b}{\|w\|} \geq d \quad y_{i}=1} \\ {\frac{w^{T} x+b}{\|w\|} \leq-d \quad y_{i}=-1}\end{array}\right.$$

因为$\|w\| d$ 是正数，其对目标函数优化无影响，这里令其为1，因此上式不等式组可简化为:

$$\left\{\begin{array}{l}{w^{T} x+b \geq 1 \quad y_{i}=1} \\ {w^{T} x+b \leq-1 \quad y_{i}=-1}\end{array}\right.$$

合并两个不等式，得到：$y_{i}\left(w^{T} x+b\right) \geq 1$

这里，我们再来看关于超平面和支持向量的图解：

<img src="https://images.bumpchicken.cn/img/20220511221210.png" width="60%">

支持向量到最大间隔超平面的距离为：

$$d=\frac{\left|w^{T} x+b\right|}{\|w\|}$$

最大化这个距离：

$$\max 2 * \frac{\left|w^{T} x+b\right|}{\|w\|}$$

对于确定的样本集来说，$\left|w^{T} x+b\right|$是个常量，因此目标函数变为：$\max \frac{2}{\|w\|}$，即$\min \frac{1}{2}\|w\|$

为了方便计算，去除根号，目标函数转化为：$\min \frac{1}{2}\|w\|^{2}$

因此得到SVM的优化问题：

$$\min \frac{1}{2}\|w\|^{2} $$

$${s.t.}\quad y_{i}\quad\left(w^{T} x_{i}+b\right) \geq 1$$


## KKT条件

上述最优化问题约束条件是不等式。如果是等式约束，可以直接用Lagrange乘数法求解，即对于下述最优化问题

$$\begin{array}{c}{\min f\left(x_{1}, x_{2}, \ldots, x_{n}\right)} \\ {\text { s.t. } \quad h_{k}\left(x_{1}, x_{2}, \ldots, x_{n}\right)=0}\end{array}$$

我们可以构造拉格朗日函数：$L(x, \lambda)=f(x)+\sum_{k=1}^{l} \lambda_{k} h_{k}(x)$，然后分别对$x$,$\lambda$求偏导，求得可能的极值点

$$\left\{\begin{array}{ll}{\frac{\partial L}{\partial x_{i}}=0} & {i=1,2, \ldots, n} \\ {\frac{\partial L}{\partial \lambda_{k}}=0} & {k=1,2, \ldots, l}\end{array}\right.$$

那么对于不等式约束条件，做法是引入一个松弛变量，然后将该松弛变量也视为待优化变量。以SVM优化问题为例：

$$\begin{aligned} \min f(w) &=\min \frac{1}{2}\|w\|^{2} \\ \text {s.t.} & g_{i}(w)=1-y_{i}\left(w^{T} x_{i}+b\right) \leq 0 \end{aligned}$$

引入松弛变量 $a_{i}^{2}$，得到新的约束条件: $h_{i}\left(w, a_{i}\right)=g_{i}(w)+a_{i}^{2}=0$，将不等式约束变为等式约束，得到新的拉格朗日函数：

$$\begin{aligned} L(w, \lambda, a) &=\frac{1}{2} f(w)+\sum_{i=1}^{n} \lambda_{i} h_{i}(w) \\ &=\frac{1}{2} f(w)+\sum_{i=1}^{n} \lambda_{i}\left[g_{i}(w)+a_{i}^{2}\right] \quad \lambda_{i} \geq 0 \end{aligned}$$

（**注意到，这里有$\lambda_{i}>=0$，在拉格朗日乘数法中，没有非负的要求，关于这里为何 $\lambda_{i}>=0$ 可以通过几何性质来证明，有兴趣的可以查阅相关资料**。）

根据等式约束条件，有：

$$\left\{\begin{array}{c}{\frac{\partial L}{\partial w_{i}}=\frac{\partial f}{\partial w_{i}}+\sum_{i=1}^{n} \lambda_{i} \frac{\partial g_{i}}{\partial w_{i}}=0} \\ {\frac{\partial L}{\partial a_{i}}=2 \lambda_{i} a_{i}=0} \\ {\frac{\partial L}{\partial \lambda_{i}}=g_{i}(w)+a_{i}^{2}=0} \\ {\lambda_{i} \geq 0}\end{array}\right.$$

第二个式子，$2\lambda_{i} a_{i}=0$，有两种情况：

1. $\lambda_{i}$ 为0，$a_{i}$不为0：由于$\lambda_{i}$为0，这时候约束$g_{i}(w)$不起作用，并且$g_{i}(w)<0$
2. $\lambda_{i}$不为0，$a_{i}$为0：这时$g_{i}(w)$起约束作用，并且$g_{i}(w)=0$

因此，方程组可转换为：

$$\left\{\begin{aligned} \frac{\partial L}{\partial w_{i}} &=\frac{\partial f}{\partial w_{i}}+\sum_{j=1}^{n} \lambda_{j} \frac{\partial g_{j}}{\partial w_{i}}=0 \\ \lambda_{i} g_{i}(w) &=0 \\ g_{i}(w) & \leq 0 \\ \lambda_{i} & \geq 0 \end{aligned}\right.$$

以上便是不等式约束优化问题的__KKT(Karush-Kuhn-Tucker)条件__，$\lambda_{i}$称为KKT乘子。从这个方程组可以得到以下讯息：

1. 对于支持向量 $g_{i}(w)=0$，$\lambda_{i}>0$即可
2. 对于非支持向量 $g_{i}(w)<0$，但要求 $\lambda_{i}=0$

## 求解SVM最优化问题

利用KKT条件，我们可以求解SVM最优化问题：

$$\min _{w} \frac{1}{2}\|w\|^{2}$$

$$\text {s.t. } \quad g_{i}(w, b)=1-y_{i} \left(w^{T} x_{i}+b\right) \quad \leq 0, \quad i=1,2, \ldots, n$$

__Step 1__: 构造拉格朗日函数

$$\begin{aligned} L(w, b, \lambda)=& \frac{1}{2}\|w\|^{2}+\sum_{i=1}^{n} \lambda_{i}\left[1-y_{i}\left(w^{T} x_{i}+b\right)\right] \\ & \text {s.t.} \quad \lambda_{i} \geq 0 \end{aligned}$$

假设目标函数最小值为p, 即$\frac{1}{2}\|w\|^{2} = p$，因为$\sum_{i=1}^{n} \lambda_{i}\left[1-y_{i}\left(w^{T} x_{i}+b\right)\right] <= 0$，即$L(w, b, \lambda) <= p$，为了找到最优的参数$\lambda$使得$L(w, b, \lambda)$接近p，问题转换为：$\max _{\lambda} L(w, b, \lambda)$，即：

$$\begin{array}{c}{\min _{w} \max _{\lambda} L(w, b, \lambda)} \\ {\text { s.t. } \quad \lambda_{i} \geq 0}\end{array}$$

__Step 2__:利用对偶性转换求解问题：

对偶问题其实就是将：

$$\begin{array}{c}{\min _{w} \max _{\lambda} L(w, b, \lambda)} \\ {\text { s.t. } \quad \lambda_{i} \geq 0}\end{array}$$

转化为：$\begin{array}{c}{\max _{\lambda} \min _{w} L(w, b, \lambda)} \\ {\text { s.t. } \quad \lambda_{i} \geq 0}\end{array}$

假设有函数$f$,我们有：

$$min \space max f >= max \space min f$$

即最大的里面挑出来最小的也要比最小的里面挑出来最大的要大，这是一种 __弱对偶关系__，而 __强对偶关系__ 是当等号成立时，即：

$$min \space max f == max \space min f$$

当$f$是凸优化问题时，等号成立，而我们之前求的KKT条件是强对偶性的 __充要条件__

因此，对$\begin{array}{c}{\max _{\lambda} \min _{w} L(w, b, \lambda)} \\ {\text { s.t. } \quad \lambda_{i} \geq 0}\end{array}$进行求解：

1）对参数$w$和$b$求偏导数：

$$\frac{\partial L}{\partial w}=w-\sum_{i=1}^{n} \lambda_{i} x_{i} y_{i}=0$$

$$\frac{\partial L}{\partial b}=\sum_{i=1}^{n} \lambda_{i} y_{i}=0$$

得到：

$$\sum_{i=1}^{n} \lambda_{i} x_{i} y_{i}=w$$

$$\sum_{i=1}^{n} \lambda_{i} y_{i}=0$$

2）将1）中求导结果代回 $L(w, b, \lambda)$中，得到：

$$\begin{aligned} L(w, b, \lambda) &=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \lambda_{i} \lambda_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{n} \lambda_{i}-\sum_{i=1}^{n} \lambda_{i} y_{i}\left(\sum_{j=1}^{n} \lambda_{j} y_{j}\left(x_{i} \cdot x_{j}\right)+b\right) \\ &=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \lambda_{i} \lambda_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{n} \lambda_{i}-\sum_{i=1}^{n} \sum_{j=1}^{n} \lambda_{i} \lambda_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{n} \lambda_{i} y_{i} b \\ &=\sum_{j=1}^{n} \lambda_{i}-\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \lambda_{i} \lambda_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right) \end{aligned}$$

3）利用SMO（Sequential Minimal Optimization）求解

$\max _{\lambda}\left[\sum_{j=1}^{n} \lambda_{i}-\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \lambda_{i} \lambda_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)\right]$  $\text { s.t. } \sum_{i=1}^{n} \lambda_{i} y_{i}=0 \quad \lambda_{i} \geq 0$

这是一个二次规划问题，问题规模正比于训练样本数，我们常用 SMO算法求解。

SMO的思路是：先固定除$\lambda_{i}$之外的参数，然后求$\lambda_{i}$上的极值。但是我们这里有约束条件$\sum_{i=1}^{n} \lambda_{i} y_{i}=0$，如果固定$\lambda_{i}$之外的参数，$\lambda_{i}$可直接由其他参数推导出来。因此这里SMO一次优化两个参数，具体步骤为：

- 选择两个需要更新的参数$\lambda_{i}$和$\lambda_{j}$，固定其他参数，于是约束变为：

  $\lambda_{i} y_{i}+\lambda_{j} y_{j}=c \quad \lambda_{i} \geq 0, \lambda_{j} \geq 0$，其中，$c=-\sum_{k \neq i, j} \lambda_{k} y_{k}$

  得到：$\lambda_{j}=\frac{c-\lambda_{i} y_{i}}{y_{j}}$

  也就是说我们可以用$\lambda_{i}$的表达式代替$\lambda_{j}$。这样就相当于把目标问题转化成了仅有一个约束条件的最优化问题，仅有的约束是$\lambda_{i}>=0$

- 对于仅有一个约束条件的最优化问题，我们完全可以在$\lambda_{i}$上对优化目标求偏导，令导数为零，从而求出变量值$\lambda_{i}$的极值$\lambda_{i_{new}}$，然后通过$\lambda_{i_{new}}$求出$\lambda_{j_{new}}$

- 多次迭代直至收敛

4）根据 ${\sum_{i=1}^{n} \lambda_{i} x_{i} y_{i}=w}$求得$w$

5）求偏移项$b$

对于$\lambda_{i}>0$，$g_{i}(w)=0$，满足这个条件的点均为支持向量，可以取任一支持向量，带入$y_{s}\left(w x_{s}+b\right)=1$，即可求得$b$

或者采取更为鲁棒的做法，取所有支持向量各计算出一个$b$，然后取均值，即按下式求取：

$$b=\frac{1}{|S|} \sum_{s \in S}\left(y_{s}-w x_{s}\right)$$

6）构造分类超平面：$w^{T} x+b=0$

分类决策函数：$f(x)=\operatorname{sign}\left(w^{T} x+b\right)$

其中，$sign(x)$是阶跃函数：

$$\operatorname{sign}(x)=\left\{\begin{array}{rl}{-1} & {x<0} \\ {0} & {x=0} \\ {1} & {x>0}\end{array}\right.$$


## 线性不可分场景

 以上我们讨论的都是线性可分的情况，实际场景中，通常遇到的数据分布都是线性不可分的。如以下场景，此场景下，求得的分类面损失将会超出我们的容忍范围。

<img src="https://images.bumpchicken.cn/img/20220511221306.png" width="60%">


**SVM的做法是将二维线性不可分样本映射到高维空间中，让样本点在高维空间线性可分**，比如下列动图演示的做法

<img src="https://images.bumpchicken.cn/img/201485427.gif" width="80%">


对于在有限维度向量空间中线性不可分的样本，我们将其映射到更高维度的向量空间里，再通过间隔最大化的方式，学习得到的支持向量机称之为**非线性 SVM。**

我们用 x 表示原来的样本点，用 $\kappa(x)$表示 $x$ 映射到特征新的特征空间后到新向量。那么优化问题可以表示为：

$$\max _{\boldsymbol{\lambda}} \sum_{i=1}^{m} \lambda{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \lambda{i} \lambda{j} y_{i} y_{j} \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)$$

$$\text {s.t.} \quad \sum_{i=1}^{n} \lambda_{i} y_{i}=0, \quad \lambda_{i} \geq 0$$

我们常用的核函数有：

- 线性核函数

  $$k\left(x_{i}, x_{j}\right)=x_{i}^{T} x_{j}$$

- 多项式核函数

  $$k\left(x_{i}, x_{j}\right)=\left(x_{i}^{T} x_{j}\right)^{d}$$

- 高斯核函数

  $$k\left(x_{i}, x_{j}\right)=\exp \left(-\frac{\left\|x_{i}-x_{j}\right\|}{2 \delta^{2}}\right)$$


 理论上高斯核函数可以将数据映射到无限维。

## 总结

SVM作为一种经典的机器学习分类方法，具有以下优点:

1. 采用核技巧之后，可以处理非线性分类/回归任务
2. 能找出对任务至关重要的关键样本（即支持向量)
3. 最终决策函数只由少数的支持向量所确定，计算的复杂性取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”

同时，也具有以下缺点：

1. 训练时间长。当采用 SMO 算法时，由于每次都需要挑选一对参数，因此时间复杂度为 O(N2)，其中 N 为训练样本的数量
2. 当采用核技巧时，如果需要存储核矩阵，则空间复杂度为 O(N2)
3. 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高



## 参考资料

1. 《机器学习》 周志华

2. [ 浅谈最优化问题中的KKT条件](https://zhuanlan.zhihu.com/p/26514613)