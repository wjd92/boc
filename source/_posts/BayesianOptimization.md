
title: 贝叶斯优化原理及应用

date: 2020-12-29 12:07:00

tags: 
  - AutoML

categories:
  - 机器学习

mathjax: true

top: 888

---
<img src="https://images.bumpchicken.cn/img/20220509122455.png">

贝叶斯优化(Bayesian Optimization, BO)是一种黑盒优化算法，用于求解表达式未知的函数极值问题。算法使用高斯过程回归对一组采样点的函数值进行概率建模，预测出任意点处函数值的概率分布，然后构造采集函数(Acquistion Function)，用于衡量每一个点值得探索(explore)的程度，求解采集函数的极值从而确定下一个采样点，最后返回这组采样点的极值作为函数的极值。

<!--more-->

## 黑盒优化问题
训练机器学习模型过程中，会有很多模型参数之外的参数，如学习率，卷积核大小等，再比如训练xgboost时，树的最大深度、采样率等参数都会影响训练结果，这些参数我们将其称为超参数。假设一组超参数组合$X=x_{1},x_{2},...,x_{n}$，存在一个未知函数 $f:x \rightarrow \mathbb {R}$，我们需要在$x \in X$找到一组最佳参数组合$x^{*}$使得:

$$x^{*} = \underset{x\in X}{argmin} f(x) $$

当$f$是凸函数并且定义域也是凸的时候，可以通过凸优化手段(梯度下降、L_BFGS等)来求解。但是超参数优化属于黑盒优化问题，$f$不一定是凸函数，并且是未知的，在优化过程中只能得到函数的输入和输出，不能获取目标函数的表达式和梯度信息，这里的$f$通常还是计算代价非常昂贵的函数，因此优化过程会比较困难，尤其是当超参数数量大的情况。常用的超参数优化方法有网格搜索(Grid Search)，随机搜索(Random Search)，遗传算法（粒子群优化、模拟退火等）以及本文要介绍的贝叶斯优化方法。

下面介绍两种最基本的超参调优方法: 网格搜索法和随机搜索法

- 网格搜索法

  网格搜索法搜索一组离散的取值情况，得到最优参数值。如果是连续型的超参数，则需要对其定义域进行网格划分，然后选取典型值计算。网格搜索法本质上是一种穷举法，对待调优参数进行全排列组合，逐一计算$f$，然后选取最小的$f$时的参数组合，如下代码所示，给定参数候选项，我们可以列出所有的参数组合
```python
from itertools import product
tuning_params = {'a':[1,2,3], 'b':[4,5]}                     # 待优化参数可选项
for conf in product(*tuning_params.values()):
    print({k:v for k,v in zip(tuning_params.keys(), conf)})  # 生成参数组合
```
输出:
```python
{'a': 1, 'b': 4}
{'a': 1, 'b': 5}
{'a': 2, 'b': 4}
{'a': 2, 'b': 5}
{'a': 3, 'b': 4}
{'a': 3, 'b': 5}
```
随着待调优参数增加，生成的全排列组合数量将非常巨大，计算代价过于昂贵
- 随机搜索法
  相比于网格搜索法，随机搜索的做法是将超参数随机地取某些值，设置一个最大迭代次数，比较每次迭代中不同取值算法的输出，得到最优超参数组合。而随机取值的方法也有多种不同的做法，常用的做法是采用均匀分布的随机数进行搜索，或者采用一些启发式的搜索策略（粒子群优化算法），这里不展开赘述。随机搜索并不总能找到全局最优解，但是通常认为随机搜索比网格搜索更优，其可以花费更少的计算代价得到相近的结果。

__无论是网格搜索法还是随机搜索法，每一次进行迭代计算的时候，都未曾考虑已经搜索过的空间，即搜索过的空间未对下一次搜索产生任何指导作用，因此可能存在很多无效搜索。不同于网格搜索和随机搜索法，贝叶斯优化则能够通过高斯过程回归有效利用先验的搜索空间进行下一次搜索参数的选择，能大大减少迭代次数__

## 理论准备
经典的贝叶斯优化利用高斯过程(Gaussian Process, GP)对$f$进行概率建模，在介绍贝叶斯优化之前，有必要了解一下高斯过程回归的相关知识

### 高斯过程
高斯过程用于对一组随着时间增长的随机向量进行建模，在任意时刻，某个向量的所有子向量均服从高斯分布。
假设有连续型随机变量序列$x_{1},x_{2},...,x_{T}$，如果该序列中任意数量的随机变量构成的向量$X_{t_{1}, ... ,t_{k}} = [x_{t_{1}}  \space ... \space  x_{t_{k}}]^{T}$均服从多维正态分布，则称次随机变量序列为高斯过程。

特别地，假设当前有k个随机变量$x_{1},...,x{k}$，它们服从k维正态分布$N( \mu_{k},  \sum _{k} )$，其中均值向量$N( \mu_{k},  \sum _{k} )$，协方差矩阵$\sum _{k} \in \mathbb R^{k*k}$

当加入一个新的随机变量$x_{k+1}$之后，随机向量$x_{1},x_{2},...,x_{k},x_{k+1}$服从k+1维正态分布$\mu_{k+1} \in \mathbb{R}^{k+1}$，其中均值向量$\mu_{k+1} \in \mathbb{R}^{k+1}$，协方差矩阵$\sum _{k+1} \in \mathbb R^{(k+1)*(k+1)}$

由于正态分布的积分能够得到解析解，因此可以方便地得到边缘概率于条件概率。

### 高斯过程回归

机器学习中，算法通常是根据输入值$x$预测出一个最佳输出值$y$，用于分类或回归。某些情况下我们需要的不是预测出一个函数值，而是给出这个函数值的后验概率分布$p(y|x)$。对于实际应用问题，一般是给定一组样本点$x_{i}, \space i=1,…,l$，基于此拟合出一个假设函数，给定输入值$x$，预测其标签值或者后验概率$p(y|x)$，高斯过程回归对应后者。

高斯过程回归(Gaussian Process Regression, GPR)对表达式未知的函数的一组函数值进行概率建模，给出函数值的概率分布。嘉定给定某些点$x_{i}, i= 1,…,t$，以及在这些点处的函数值$f(x_{i})$，GPR能够根据这些点，拟合该未知函数，那么对于任意给定的$x$，就可以预测出$f(x)$，并且能够给出预测结果的置信度。

GPR假设黑盒函数在各个点处的函数值$f(x)$都是随机变量，它们构成的随机向量服从多维正态分布。假设有t个采样点$x_{1},…,x_{t}$，在这些点处的函数值构成向量：

$$f(x_{1:t}) = [f(x_{1} \space ... \space f(x_{t})]$$

GPR假设此向量服从t维正态分布：

$$f(x_{1:t}) \sim N(\mu(x_{1:t}), \sum(x_{1:t},x_{1:t}))$$

其中，$\mu(x_{1:t})=[\mu(x_{1}),…,\mu(x_{t})]$是高斯分布的均值向量，$\sum(x_{1:t},x_{1:t})$是协方差矩阵

$$\left[\begin{array}{ccc}
\operatorname{cov}\left(\mathbf{x}_{1}, \mathbf{x}_{1}\right) & \ldots & \operatorname{cov}\left(\mathbf{x}_{1}, \mathbf{x}_{t}\right) \\
\cdots & \ldots & \ldots \\
\operatorname{cov}\left(\mathbf{x}_{t}, \mathbf{x}_{1}\right) & \ldots & \operatorname{cov}\left(\mathbf{x}_{t}, \mathbf{x}_{t}\right)
\end{array}\right]=\left[\begin{array}{ccc}
k\left(\mathbf{x}_{1}, \mathbf{x}_{1}\right) & \ldots & k\left(\mathbf{x}_{1}, \mathbf{x}_{t}\right) \\
\ldots & \ldots & \ldots \\
k\left(\mathbf{x}_{t}, \mathbf{x}_{1}\right) & \ldots & k\left(\mathbf{x}_{t}, \mathbf{x}_{t}\right)
\end{array}\right]$$

问题的关键是如何根据样本值计算出正态分布的均值向量和协方差矩阵，均值向量是通过均值函数$\mu(x)$根据每个采样点x计算构造的，可简单令$\mu(x)=c$，或者将均值设置为0，因为即使均值设置为常数，由于有方差的作用，依然能够对数据进行有效建模。

协方差通过核函数$k(x,x^{'})$计算得到，也称为协方差函数，协方差函数需要满足以下要求：

1. 距离相近的样本点$x$和$x^{'}$之间有更大的正协方差值，因为相近的两个点的函数值有更强的相关性
2. 保证协方差矩阵是对称半正定矩阵

常用的是高斯核和Matern核，高斯核定义为：

$$k\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right)=\alpha_{0} \exp \left(-\frac{1}{2 \sigma^{2}}\left\|\mathbf{x}_{1}-\mathbf{x}_{2}\right\|^{2}\right)$$

Matern核定义为:

$$k\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right)=\frac{2^{1-v}}{\Gamma(v)}\left(\sqrt{2 v}\left\|\mathbf{x}_{1}-\mathbf{x}_{2}\right\|\right)^{v} K_{v}\left(\sqrt{2 v}\left\|\mathbf{x}_{1}-\mathbf{x}_{2}\right\|\right)$$

其中$\Gamma$是伽马函数，$K_{v}$是贝塞尔函数(Bessel function)，$v$是人工设定的正参数。用核函数计算任意两点之间的核函数值，得到核函数矩阵$K$作为协方差矩阵的估计值：

$$\mathbf{K}=\left[\begin{array}{ccc}
k\left(\mathbf{x}_{1}, \mathbf{x}_{1}\right) & \ldots & k\left(\mathbf{x}_{1}, \mathbf{x}_{t}\right) \\
\ldots & \ldots & \ldots \\
k\left(\mathbf{x}_{t}, \mathbf{x}_{1}\right) & \ldots & k\left(\mathbf{x}_{t}, \mathbf{x}_{t}\right)
\end{array}\right]$$

在计算出均值向量和协方差矩阵之后，可以根据此多维正态分布预测$f(x)$在任意点处的概率分布。假设已经得到了一组样本$X_{1:t}$，以及对应的函数值$f(x_{1:t})$，如果要预测新的点$x$的函数值$f(x)$的期望$\mu(x)$和方差$\sigma^{2}(x)$，令$x_{t+1}=x$，加入该点后，$f(x_{1:t+1})$服从$t+1$维正态分布，即：

$$\left[\begin{array}{c}
f\left(\mathbf{x}_{1: t}\right) \\
f\left(\mathbf{x}_{t+1}\right)
\end{array}\right] \sim N\left(\left[\begin{array}{c}
\mu\left(\mathbf{x}_{1: t}\right) \\
\mu\left(\mathbf{x}_{t+1}\right)
\end{array}\right],\left[\begin{array}{cc}
\mathbf{K} & \mathbf{k} \\
\mathbf{k}^{\mathrm{T}} & k\left(\mathbf{x}_{t+1}, \mathbf{x}_{t+1}\right)
\end{array}\right]\right)$$

在已知$f(x_{1:t})$的情况下，$f(x_{t+1})$服从一维正态分布，即：

$$f\left(\mathbf{x}_{t+1}\right) \mid f\left(\mathbf{x}_{1: t}\right) \sim N\left(\mu, \sigma^{2}\right)$$

可以计算出对应的均值和方差，公式如下：

$$\begin{array}{l}
\mu=\mathbf{k}^{\mathrm{T}} \mathbf{K}^{-1}\left(f\left(\mathbf{x}_{1: t}\right)-\mu\left(\mathbf{x}_{1: t}\right)\right)+\mu\left(\mathbf{x}_{t+1}\right) \\
\sigma^{2}=k\left(\mathbf{x}_{t+1}, \mathbf{x}_{t+1}\right)-\mathbf{k}^{\mathrm{T}} \mathbf{K}^{-1} \mathbf{k}
\end{array}$$

计算均值利用了已有采样点处函数值$f(x_{1:t})$，方差只与协方差值有关，与$f(x_{1:t})$无关

### GPR代码实现

- 定义高斯核

```python
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    X1: Array of m points (m x d).
    X2: Array of n points (n x d).
    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
```

- 计算均值和协方差矩阵

```python
def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    根据先验数据点计算均值向量和协方差矩阵
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = np.linalg.inv(K)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)      # 均值向量, 注意均值函数被设置为 0
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)  # 协方差矩阵
    return mu_s, cov_s
```

- GPR拟合效果绘制

```Python
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    # 定义gp绘图函数
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))   # 1.96倍标准差对应95%置信度区间

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
      plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
      if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
        plt.legend()

X = np.arange(-5, 5, 0.2).reshape(-1, 1)
X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
Y_train = np.sin(X_train)

# 计算均值向量和协方差矩阵
mu_s, cov_s = posterior_predictive(X, X_train, Y_train)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 1)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
```

结果如下图所示：

<img src="https://images.bumpchicken.cn/img/20220510004441.png" width="50%" height="50%">

其中，红色叉代表观测值，蓝色线代表均值，浅蓝色区域代表95%置信区间。

## 贝叶斯优化原理

#### 基本过程

以下是贝叶斯优化的过程

<img src="https://images.bumpchicken.cn/img/20220510004715.png">

有两个主要组成部分：

1. GPR。根据观测点构建高斯过程回归模型，该模型能求取任意点处的函数值及后验概率。
2. 构造采集函数（acquisition function），用于决定本次迭代在哪个点处进行采样。

算法首先初始化$n_{0}$个点，设定最大迭代次数$N$，开始循环求解，每次增加一个点，寻找下一个点时根据已经找到的$n$个候选解建立高斯回归模型，通过这个模型能得到任意点处的函数值的后验概率。然后根据后验概率构造采集函数，寻找采集函数的极大值点作为下一个搜索点，以此循环，直到达到最大迭代次数，返回$N$个解中的极大值作为最优解。

采集函数的选择有很多种，最常用的是期望改进（Expected Improvement，EI),下一节介绍下EI的原理。

#### 采集函数

假设已经搜索了n个点，这些点中的函数极大值记为

$$f_{n}^{*} = max(f(x_{1},...,f(x_{n}))$$

考虑下一个搜索点，计算该点处的函数值$f(x)$，如果$f(x)>=f_{n}^{*}$，则这$n+1$个点处的函数极大值为$f(x)$，否则为$f_{n}^{*}$

加入这个新的点后，函数值的改进可以记为

$$e^{+} = max(0, f(x) - f_{n}^{*})$$

我们的目标是找到使得上面的改进值最大的$x$，但是该点的函数值在我们找到$x$是多少前又是未知的，幸运的是我们知道$f(x)$的概率分布，因此我们可以计算在所有x处的改进值的数学期望，然后选择期望最大的$x$作为下一个搜索点。定义期望改进（EI）函数如下：

$$EI_{n}(x) = E_{n}[max(0, f(x) - f_{n}^{*})]$$

令$z=f(x)$，则有：

$$\begin{aligned}
\mathrm{EI}_{n}(\mathbf{x}) &=\int_{-\infty}^{+\infty}\left (max(0, z-f_{n}^{*})\right) \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{(z-\mu)^{2}}{2 \sigma^{2}}\right) d z \\
&=\int_{f_{n}^{*}}^{+\infty}\left(z-f_{n}^{*}\right) \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{(z-\mu)^{2}}{2 \sigma^{2}}\right) d z
\end{aligned}$$

换元法，得到：

$$\\
\begin{array}{c}
\int_{f_{n}^{*}}^{+\infty}\left(z-f_{n}^{*}\right) \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{(z-\mu)^{2}}{2 \sigma^{2}}\right) d z
 \\
=\left(\mu-f_{n}^{*}\right)\left(1-\Phi\left(\left(f_{n}^{*}-\mu\right) / \sigma\right)\right)+\sigma \varphi\left(\left(f_{n}^{*}-\mu\right) / \sigma\right)
\end{array}$$

其中，$\varphi(x)$是标准正态分布的概率密度函数，$\phi(x)$是是标准正态分布的分布函数

我们的目标是求EI的极值获取下一个采样点，即

$$x_{n+1} = argmax EI_{n}(x)$$

现在目标函数已知，且能得到目标函数的一阶导数和二阶导数，可以通过梯度下降法或L-BFGS求解极值，这里不再展开叙述



## 贝叶斯优化应用

BO有许多开源的实现，scikit-optimize 以及 参考资料4 [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)都封装了BO，这里我们采用参考资料4中封装的BO进行演示。

1. 直接用pip安装BayesianOptimization

```shell
pip install bayesian-optimization
```

2. 定义黑盒函数

```python
def black_box_function(x, y):
    """
    x,y 均是待调优参数
    """
    return -x ** 2 - (y - 1) ** 2 + 1   
```

3. 初始化BO

```python
from bayes_opt import BayesianOptimization

pbounds = {'x': (2, 4), 'y': (-3, 3)}      # 设定x, y 调参范围

# 初始化bo
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)
```

4. 进行迭代

```python
optimizer.maximize(
    init_points=2,          # 初始解个数
    n_iter=20,               # 迭代次数
)
print(optimizer.max)       # 输出最大值及对应的参数组合
```

输出：

<img src="https://images.bumpchicken.cn/img/20220511172849.png" width="80%" height="80%">

函数$f(x,y) = -x^{2} - (y-1)^{2} + 1$，当$x\in[2,4]$，$y\in[-3,3]$时，很显然，当$x=2,y=1$时能取到最大值，BO给出的解已经相当接近最优解

运行了多次，BO给出的解非常稳健，如下所示：

<img src="https://images.bumpchicken.cn/img/20220511173250.png">

## 参考资料

1. 《机器学习 原理、算法与应用》 雷明著

2. Frazier P I. A tutorial on bayesian optimization[J]. arXiv preprint arXiv:1807.02811, 2018.

3. https://github.com/krasserm/bayesian-machine-learning

4. https://github.com/fmfn/BayesianOptimization

5. https://github.com/scikit-optimize/scikit-optimize
