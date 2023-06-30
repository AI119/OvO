Bagging算法采用平均方法的思想，也称为汇聚法(Bootstrap Aggregating)。算法过程如下：
1. 在原始数据集上进行随机抽样（抽样可以是放回抽样与不放回抽样）。 
2. 使用得到的随机子集来训练评估器（基本评估器）。 
3. 重复步骤1与步骤2若干次，会得到n个基本评估器。 
4. 将$n$个评估器进行组合，根据多数投票（分类）或者求均值（回归）的方式来统计最终的结果（平均方法）。
![[Pasted image 20230605170056.png]]

### 优势
---------
bagging方法通过随机抽样来构建原始数据集的子集，来训练不同的基本评估器，然后再将多个基本评 估器进行组合来预测结果，这样可以有效减小基本评估器的方差。因此，通过bagging方法，就可以非 常便捷的对基本评估器进行改进，而无需去修改基本评估器底层的实现。 
因为bagging方法可以有效的降低过拟合，因此，bagging方法适用于强大而复杂的模型（例如，完全 生长的决策树）。

<font color="05B3F8" size="5.4">随机选择部分数据集而不选择全部数据集的原因是：每一个模型都有机会接触到数据集，可以增强基本模型的多样性，增强模型的泛化能力，不容易产生过拟合。</font>

## 程序示例

[[生成数据集]]

### 分类数据集示例

``` python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, n_classes=3, random_state=0,flip_y=0.25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
print(f"决策树正确率{tree.score(X_train, y_train)}")
print(tree.score(X_test, y_test))
# estimator：指定基本评估器。即bagging算法所组合的评估器。
# n_estimators：基本评估器的数量。有多少个评估器，就会进行多少次随机采样，产生多少个原始数据集的子集。
# max_samples：每次随机采样的样本数量。该参数可以是int类型或float类型。如果是int类型，则指定采样的样本数量。
# 如果是float类型，则指定采样占原始数据集的比例。
# max_features：训练每个评估器的特征数量。可以是int类型或float类型。
# bootstrap：指定是否进行放回抽样。默认为True。
# bootstrap_features：指定对特征是否进行重复抽取。默认为False。
bag = BaggingClassifier(
    estimator=tree, bootstrap=True,
    bootstrap_features=True, max_features=0.6,
    max_samples=0.68,n_estimators=100
)
bag.fit(X_train,y_train)
print(bag.score(X_train, y_train))
print(bag.score(X_test, y_test))
```

### 回归示例
``` python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

X, y = make_regression(n_samples=2000, n_features=20, n_informative=2, noise=30, random_state=12, bias=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=199)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("线性回归 R^2值：")
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

bag = BaggingRegressor(
    estimator=lr, n_estimators=100, max_samples=0.9,
    max_features=0.9
)
#选择一部分特征训练决策树，防止过拟合。
bag.fit(X_train, y_train)
print("bagging R^2值：")
print(bag.score(X_train, y_train))
print(bag.score(X_test, y_test))


```

bagging适合集成一些强模型，比如随机树，弱模型，比如线性回归可能不会产生太多改进。