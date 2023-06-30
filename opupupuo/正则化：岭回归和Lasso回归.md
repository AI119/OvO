[[欠拟合与过拟合]]
在线性回归中，模型过于复杂，通常表现为模型参数(绝对值)过大时，容易出现过拟合现象，可以通过正则化来降低过拟合的程度。正则化，就是通过在损失函数中加入关于权重的惩罚项，进而限制模型的参数过大，从而减低过拟合，增加的惩罚项，我们也称作正则项。

------
## *常用的正则化*
在线性回归中，模型过于复杂，通常表现为模型的参数过大（指绝对值过大），即如果模型的参数过大，就容易出现 过拟合现象。我们可以通过正则化来降低过拟合的程度。正则化，就是通过在损失函数中加入关于权重的惩罚项，进 而限制模型的参数过大，从而减低过拟合，增加的惩罚项，我们也称作正则项。 根据正则项的不同，我们可以将正则化分为如下几种：
+ L2正则化
+ L1正则化 
+ Elastic Net
--------

## *L2正则化*
L2正则化是最常使用的正则化，将所有权重的平方和作为正则项。使用L2正则的线性回归模型称为Ridge回归（岭回归）。加入L2正则化的损失函数为：
$J\left(w\right)=\frac12{\textstyle\sum_{i=1}^m}{(y^{(i)}-\widehat y^{(i)})}^2+\alpha{\textstyle\sum_{j=1}^n}w_j^2$ 
  + m：样本数量
  + n：特征数量
  + $\alpha$：惩罚系数（$\alpha$ > 0）

---------

## *L1正则化*
L1正则化使用所有权重的绝对值和作为正则项。使用L1正则的线性回归模型称为Lasso回归（Least Absolute Shrinkage and Selection Operator— —最小绝对值收缩与选择因子）。
$j\left(w\right)=\frac12\sum_{i=1}^m\left(y^{\left(i\right)}-\widehat y^{\left(i\right)}\right)^2+\alpha\sum_{j=1}^n\left|w_j\right|$

-----

## *Elastic Net*
Elastic Net（弹性网络），同时将绝对值和与平方和作为正则项，是L1正则化与L2正则化之间的一个折中。使用该正 则项的线性回归模型称为Elastic Net算法。

$j\left(w\right)=\frac12\sum_{i=1}^m\left(y^{\left(i\right)}-\widehat y^{\left(i\right)}\right)^2+\alpha(p\sum_{j=1}^n\left|w_j\right|+(1-p){\textstyle\sum_{j=1}^n}w_j^2)$

+ $p$ : L1正则化的比重$（0 <= p <= 1 )$

-----

## *正则化说明*

<font face="微软雅黑" size=5> Lasso </font>

优点： 

+ Lasso的正则化有助于减少模型的过度拟合，从而实现更好的泛化。 
+ Lasso可以通过强制某些系数恰好为零来执行特征选择，这有助于识别数据集中最重要的特征。 
+ 由于Lasso导致模型稀疏，因此，仅使用有限数量的非零系数来解释模型变得更加容易。 

缺点： 

+ 在多个特征高度相关时，Lasso可能只选择其中一个并丢弃其他特征，这可能导致性能不佳。 
+ 如果参数选择不当，可能会导致欠拟合或过拟合。

<font face="微软雅黑" size=5> Ridge </font>

优点：

+ Ridge的正则化还有助于减少模型的过度拟合，从而实现更好的泛化。
+ 当多个特征高度相关时，Ridge可以为这些特征分配系数值来处理。
+ Ridge对数据中的噪声更加稳健，因此会获得更好的性能。
 
 缺点：
 + Ridge不会产生稀疏解，相对于Lasso，会使模型更难解释。
 + 如果参数选择不当，可能会导致欠拟合或过拟合。

|  对比内容  |    Lasso    |    Ridge   |
| :--: | :--:| :--:|
|正则项|权重绝对值|权重平方|
|特征选择（稀疏解）|是|否|
|模型可解释性|容易|不容易|
|处理高度相关特征的能力|较差|较好|
|噪声数据|较差|较好|


## 通过Lasso实现特征选择
-----
以模拟生成的数据为例，来演示通过Lasso结合SelectFromModel实现特征选择。
``` python
#引入Lasso回归  
from sklearn.linear_model import Lasso  
#生成回归问题的合成数据集。  
from sklearn.datasets import make_regression  
  
X, y= make_regression(n_samples=10, n_features=10, coef=False, random_state=1, bias=3.5, noise=1)  
lasso = Lasso(alpha=1)  
lasso.fit(X, y)  
lasso.coef_
```

``` python
#该类用于特征选择
from sklearn.feature_selection import SelectFromModel 
# estimator：评估器，即SelectFromModel类要进行特征选择的模型。 
# threshold：阈值，当特征权重小于阈值时，丢弃该特征。 
# prefit：传入的评估器（estimator参数）是否已经训练过了。默认为False。 
sfm = SelectFromModel(estimator=lasso, threshold=1e-5, prefit=True) 
X_transform = sfm.transform(X) 
print(X_transform[:3]) 
# 返回布尔数组，用来表示是否选择对应的特征，True为选择，False为丢弃。
print(sfm.get_support())
```

## Ridge实例化
--------
``` python
from sklearn.datasets import make_regression 
from sklearn.linear_model import Ridge 
# 注意：当坐标轴使用对数比例后，这里需要改成英文字体，否则无法正常显示。 
plt.rcParams["font.family"] = "serif" 
# 创建回归数据集。 
# n_samples：样本数量。 
# n_features：特征数量。 
# coef：是否返回权重。默认为False。 
# random_state：随机种子。 # bias：偏置。
# noise：增加的噪声干扰，值越大，干扰越大。
X, y, w = make_regression(n_samples=10, n_features=10, coef=True, random_state=1, bias=3.5, noise=0.0) 
alphas = np.logspace(-4, 4, 200) 
# 定义列表，用来保存在不同alpha取值下，模型最优的权重（w）值。 
coefs = [] 
# 创建岭回归对象。 
ridge = Ridge() 
for a in alphas: 
	# alpha：惩罚力度，值越大，惩罚力度越大。
	ridge.set_params(alpha=a) 
	ridge.fit(X, y) 
	# 将每个alpha取值下，Ridge回归拟合的最佳解（w）加入到列表中。 
	coefs.append(ridge.coef_) 
# gca get current axes 获取当前的绘图对象。 
ax = plt.gca() 
# 当y是二维数组时，每一列会认为是一个单独的数据集。 
ax.plot(alphas, coefs) 
# 设置x轴的比例。（对数比例） 
ax.set_xscale("log") 
# 设置x轴的标签。 
ax.set_xlabel("alpha") 
# 设置y轴的标签。 
ax.set_ylabel("weights")
```
