+ 函数极值求解
	+ 无条件极值
	+ 条件极值
		+ 拉格朗日乘数法
		+ KKT条件
+ SVM分类思想

首先回顾[[逻辑回归]]算法
+ 分类原理
+ 损失函数
+ 决策函数
+ 决策边界

## 函数极值

### 极值点与驻点
+ 在无附加条件的情况下，函数的极值点与驻点之间没有特定的关系
	+ 驻点并不一定是极值点，极值点也不一定是驻点。
	+ $x^{3}$在0处为函数的驻点，但不是极值点。
	+ $|x|$在0处为函数的极值点，但不是驻点。
+ 如果函数是凸函数，则驻点是函数取得极值点的充要条件。
	+ 驻点是极值点。
	+ 极值点也是驻点

### 极值分类
在求解函数极值时，可以分为以下两种： 
1. 无条件极值。 
2. 条件极值。 
	+ 等式约束条件。 
	+ 不等式约束条件。
	+ 
### 无条件极值
对于函数的自变量，除了限定在函数的定义域以内，没有其他限定条件，这样的极值问题称为**无条件极值**。

如：
给定函数$f(x)$:
$$f(x)=x^2-4*x+4$$
$f(x)$是凸函数，具有唯一的极小值，定义域$(-\infty,+\infty)$根据之前的介绍，凸函数的极值一定在驻点出取得，我们可以对$f(x)$求导，令导函数为0.
$$f^{\prime}(x)=2*x-4=0$$
结果为$$x=2$$

## 条件极值（一）

对自变量除了限定在函数的定义域以内，还有其他的限制条件，这种极值问题称为**条件极值**。例如，某厂家 设计长方体容器，在表面积确定的情况下，如何才能使得体积最大？此时，我们可以通过**拉格朗日乘数法**来 求解这种条件极值问题。

### 拉格朗日乘数法

我们以极小值为例，在求解函数$f(x)$在附加条件$g(x)=0$下的极值点：
$$\begin{gathered}\underset{x}{\text{min}}f(x) \\
s.t.g_i(x)=0 , i=1,2,\ldots,m 
\end{gathered}$$
可以引入辅助函数：
这个函数由原函数和附加条件组合而来

$$L(x,\alpha)=f(x)+\sum_{i=1}^m\alpha_i g_i(x)$$
其中，$L(x)$称为拉格朗日函数，$\alpha$称为拉格朗日乘子。


### 极值求解方法
然后，我们求$L(x)$对$x$与$\alpha$的一阶偏导数，令偏导数为0，如下：
<font color="ED5B8D" size="5.7" >x和α是有多个的</font>
$$\left\{\begin{aligned}\bigtriangledown_x L(x,\alpha)&=0\\ 
\bigtriangledown_\alpha L(x,\alpha)&=0\end{aligned}\right.$$
上述方程组求解处$x$与$\alpha$，$x$就是函数$f(x)$在附加条件$g(x)=0$下可能的极值点，如果函数是凸函数，则$x$一定是极值点。

### 拉格朗日原理解析

<big>单个约束条件</big>

如下图所示
![[Pasted image 20230603124252.png]]
蓝色为函数$f(x)$的等高线，红色为约束条件$g(x)=c$
当无条件约束时，$f(x)$函数的极值点在图的中心，有条件约束时，极值点在约束条件$g(x)$上。
要注意的是极值点一定不在两个函数相交的位置上，极值点一定会<font color=05B3F8 size=5.4>在两个函数的切点上。</font>
<font color="14EBD3" size="6">梯度的方向一定是垂直于等高线的</font>

当两个函数的梯度共线时，他们的梯度向量是成比例的，如：
$$\bigtriangledown_xf(x) = \alpha \bigtriangledown_x g(x)$$在切点位置（极值点）有什么样的规律？
根据梯度一定垂直于等高线原则，可以得到，在切点位置（极值点），函数f与函数g的梯度一定是共线的（平行）。

向量（梯度）共线，则两个向量（梯度）是成比例的，因此，可以得出$$
\begin{aligned}
& L(x, \alpha)=f(x)+\alpha g(x) \\
& \left\{\begin{array} { l } 
{ \nabla _ { x } f ( x ) = \alpha \nabla _ { x } g ( x ) } \\
{ g ( x ) = 0 }
\end{array} \Leftrightarrow \left\{\begin{array}{l}
\nabla_x L(x, \alpha)=0 \\
\nabla_\alpha L(x, \alpha)=0
\end{array}\right.\right.
\end{aligned}
$$

f的负梯度一定是原理约束区域的。
因为沿着负梯度方向一定可以让函数f的值更小。
如果f的负梯度不是远离约束区域，而是靠近约束区域，这意味着，一定可以沿着函数f负梯度方向继续前进，使得函数f的值更小。

<big>多个约束条件</big>

在多个约束条件下，相交点的位置上，函数$f$的梯度并不要求与其中任何一个约束函数的梯度平行，而是要求 是这些约束函数梯度的线性组合。

![[Pasted image 20230603154940.png]]

$$
\bigtriangledown_xf(x)=\sum_{i=1}^m\alpha_i\bigtriangledown g_i(x)
$$
求每个条件函数的偏导，所以有如下结论$$
\begin{array}{l}L(x,\alpha)=f(x)+\sum_{i=1}^m\alpha_i g_i(x)\\ \left\{\begin{array}{l}\bigtriangledown_xf(x)=\sum_{i=1}^m\alpha_i\triangledown_x g_i(x)\\ g_i(x)=0,i=1,2,\ldots,m\end{array}\right.\Leftrightarrow\left\{\begin{array}{l}\bigtriangledown_x L(x,\alpha)=0\\ \bigtriangledown_\alpha L(x,\alpha)=0\end{array}\right.\end{array}
$$

## 条件极值（二）

条件极值也可以是不等式约束条件。 我们依然以极小值为例，在求解函数$f(x)$在附加条件$g(x)=0$下的极值点：

$$\begin{gathered}\underset{x}{\text{min}}f(x) \\
s.t.g_i(x)\leqslant0 , i=1,2,\ldots,m 
\end{gathered}$$

同样构造拉格朗日函数，格式为：
$$
L(x,\beta)=f(x)+\sum_{i=1}^m\beta_ig_i(x)
$$


### 松弛互补条件

![[Pasted image 20230604114149.png]]

由上图可知，可行解需要在限定函数$g(x)\leqslant0$的区域内，这包含两种情况：
1. 可行解落在函数$g(x)<0$的区域内时，此时相当于没有约束条件（$\beta=0$），直接求$f$最小值即可。
2. 可行解落在函数$g(x)=0$的区域内时，此时相当于没有约束条件（$\beta\neq0$）
![[Pasted image 20230604115005.png]]

上述两点可得出一个结论，可行解必须满足如下条件：
$$\beta g(x)=0$$

## 逻辑回归算法回顾
### 决策函数

之前，我们学习过逻辑回归等分类算法，也熟知该算法的分类原理。我们以二分类为例，对于逻辑回归来 说，通过决策函数 （返回值为 ）来实现分类，如下：
$$
\begin{gathered}
F(X)=\vec{w}^T \vec{x} \\
\begin{cases}F(X)>0 & 1 \\
F(X) \leqslant 0 & 0\end{cases}
\end{gathered}
$$

### 决策边界
从空间几何的角度来讲，决策函数确定决策边界，将样本划分在决策边界的两侧，从而实现分类的效果。在 逻辑回归算法中，决策边界的方程为：
$$
F(X)=0 \quad \Rightarrow \quad \vec{w}^T \vec{x}=0
$$
![[Pasted image 20230531152037.png]]

## 损失函数
## 点到直线的距离

在二维空间中，点 $\left(x_0, y_0\right)$ 到直线 $A x+B y+C=0$ 的距离公式为:
$$
d=\frac{\left|A x_0+B y_0+C\right|}{\sqrt{A^2+B^2}}
$$
+ $d$：点到直线的距离

## SVM算法介绍
![[Pasted image 20230606103922.png]]

SVM（Support Vector Machine），即支持向量机，可用于分类，回归或异常值检测等任务。该算法最早用 于解决二分类任务，其思想为在高维空间中构建一个（或一组）超平面（决策边界），使得在正确分类的同 时，能够让离超平面最近的样本，到超平面的距离尽可能的远（算法优化目标）。而距离超平面最近的样 本，我们称为支持向量。 支持向量到超平面的距离较远，可以使得模型具有更好的泛化能力，降低过拟合的风险。

<font color="05B3F8" size="5.4">SVM算法分类的思想是让分类的容错空间尽可能的大，也就是说让样本与决策边界尽可能的远</font>

## SVM程序实例

导入相关的库，并生成随机数据集：

``` python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "SimHei" 
plt.rcParams["axes.unicode_minus"] = False
from sklearn.datasets import make_classification 
# 生成分类的数据，这里为了演示SVM的支持向量，有意将分类数据简化。 
# n_redundant：数据集中冗余特征的数量，冗余特征会增加数据集的复杂性。 
# n_clusters_per_class：每个类别中的簇数，增加每个类别的簇数会使数据集更复杂。 
# class_sep：类别（簇）之间的距离，较大的值，分类更加容易。 
X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_classes=2, class_sep=2.8, n_clusters_per_class=1, flip_y=0, random_state=11) 
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="r", marker="o", label="类别0") 
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="g", marker="v", label="类别1") 
plt.legend()
```

![[Pasted image 20230606105420.png]]

拟合数据并且绘制决策边界

``` python
from matplotlib.colors import ListedColormap

from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.svm import SVC

def plot_decision_boundary(model, X, y):

    color = ["r", "g", "b"]

    marker = ["o", "v", "x"]

    class_label = np.unique(y)

    cmap = ListedColormap(color[: len(class_label)])

    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict",

alpha=0.5, cmap=cmap,

grid_resolution=300)

    for i, class_ in enumerate(class_label):

        plt.scatter(x=X[y == class_, 0], y=X[y == class_, 1],

            c=cmap.colors[i], label=class_, marker=marker[i])

    plt.legend()

# C：正则化参数。C值越大，对误分类的惩罚增大，使得模型在训练集上的准确率更高，但泛化能力较弱。

# C值较小时，对误分类的惩罚减小，允许一定程度的容错，泛化能力较强。

# kernel：核函数。用于将输入数据映射到更高维空间的方法，以便在这个高维空间中找到一个更好的分类边界。

# linear：线性核函数。

# poly：多项式核函数。

# rbf：径向基核函数（Radial Basis Function, RBF），也称为高斯核函数，适用于大多数问题，也是默认值。

# sigmoid：Sigmoid 核函数。

# precomputed：自行完成数据的映射，而不是让模型去计算。

# degree：多项式的阶数，默认值为3。只有在核函数为poly时有意义。

# gamma：rbf，poly与sigmoid核系数的系数，控制支持向量的影响范围。

# 较高的gamma值使模型对单个数据点更敏感，从而导致更灵活（非线性）的决策边界，这可能会导致过度拟合。

# 较低的gamma值会产生更平滑的决策边界，模型会变得简单，可能会导致欠拟合。

# coef0：poly与sigmoid核函数的常数项。

svc = SVC(kernel="linear")

svc.fit(X, y)

plot_decision_boundary(svc, X, y)
```

![[Pasted image 20230606105433.png]]

``` python 
def plot_decision_boundary2(model, X, y):
    color = ["r", "g", "b"]
    marker = ["o", "v", "x"]
    class_label = np.unique(y)
    cmap = ListedColormap(color[: len(class_label)])
    # plot_method：用于绘制图像使用的方法。默认为contourf。
    # response_method：decision_function表示使用决策函数的值。
    # levels，colors，linestyles：传递给plot_method（contour）的参数。
    DecisionBoundaryDisplay.from_estimator(model, X, plot_method="contour",
response_method="decision_function",
grid_resolution=300, levels=[-1, 0,
1], colors=["r", "b", "g"],
linestyles='--')
    for i, class_ in enumerate(class_label):
        plt.scatter(x=X[y == class_, 0], y=X[y == class_, 1],
            c=cmap.colors[i], label=class_, marker=marker[i])
    plt.legend()
plot_decision_boundary2(svc, X, y)
```

![[Pasted image 20230606105626.png]]
