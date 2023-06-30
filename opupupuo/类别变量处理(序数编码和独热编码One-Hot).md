``` python
#OrdinalEncoder用于对特征编码，LabelEncoder用于对目标值（标签）编码。 
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
```
只是为了区分特征之间的不同时，可以使用类别变量编码。使用encoder时，可以在未知的数据集上也可以标记不同的特征。
要注意的是，`OrdinalEncoder`用于标记X，`LabelEncoder`用来标记y。

``` python
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder 
X, y = data.iloc[:10, :-1], data.iloc[:10, -1] 
oe = OrdinalEncoder() 
X = oe.fit_transform(X) 
print(X) 
# 输出每个特征的类别信息。 
print(oe.categories_)
```
上述代码中，每个特征的类别信息是按照输出后的索引标记的。

``` python
le = LabelEncoder() 
y = le.fit_transform(y) 
print(y) 
# 输出标签的类别信息。 
print(le.classes_)
```


在电信用户流失数据集中，我们分别使用了序数编码和One-Hot编码对样本特征进行标记。

### 序数编码

  
序数编码（Ordinal Encoding）和独热编码（One-Hot Encoding）是两种常用的特征编码方法，用于将分类变量转换为可用于机器学习算法的数值表示。它们之间有以下区别和各自的优缺点：

1. 序数编码（Ordinal Encoding）：
    
    - 定义：序数编码将每个不同的类别映射到一个整数值，根据类别的顺序分配数值，使得较大的数值表示较大的类别。
    - 优点：
        - 保留了类别之间的顺序信息，适用于具有自然顺序关系的分类变量。
        - 生成的编码是数值型数据，可以直接用于大多数机器学习算法。
    - 缺点：
        - 引入了偏好或权重，因为类别之间的数值差异可能会影响某些算法的结果。
        - 在某些情况下，机器学习算法可能会错误地认为不同的类别之间存在某种顺序关系，从而引入错误的影响。

``` python
#序数编码的类。
from sklearn.preprocessing import OrdinalEncoder

#所有类别变量列表。
col = ["SeniorCitizen", "Partner", "Dependents", "MultipleLines", "OnlineSecurity", "OnlineBackup",
"DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
"PaymentMethod"]

oe = OrdinalEncoder()
X_train_ord = oe.fit_transform(X_train[col])
X_test_ord = oe.transform(X_test[col])

print(X_train_ord[:3])
#获取每个特征的序数编码明细。
print(f"每个特征的序数编码明细:{oe.categories_}")
lr = LogisticRegression()
score = cross_val_score(lr, X_train_ord, y_train, cv=5, scoring="f1")
```
![[Pasted image 20230606113510.png]]
*序数编码是按照每个特征中不同的种类进行标记，规则是按照所在列表中的索引位置进行编码*


### One-Hot编码

1. 独热编码（One-Hot Encoding）：
    
    - 定义：独热编码将每个不同的类别转换为一个二进制向量，其中只有一个元素为1，其余元素为0。每个类别对应一个向量。
    - 优点：
        - 消除了类别之间的顺序关系，避免了引入错误的权重。
        - 对于大多数机器学习算法而言，独热编码是一种更为直观和有效的表示方式。
    - 缺点：
        - 当类别数量很多时，独热编码会导致编码后的向量维度急剧增加，可能会带来维度灾难（curse of dimensionality）。
        - 在某些算法中，独热编码可能导致冗余的特征表示，增加计算复杂度。

``` python
#One-Hot编码的类。
from sklearn.preprocessing import OneHotEncoder
# sparse_output：是否输出稀疏矩阵，默认为True。稀疏矩阵能够节约内存空间，
# 但对于某些操作，不像密集矩阵的支持那样好，例如，数组拼接，某些模型训练等。
ohe = OneHotEncoder(sparse_output=False)
X_train_oh = ohe.fit_transform(X_train[col])
X_test_oh = ohe.transform(X_test[col])
print(X_train_oh[:3])
# 获取每个输入特征的名称。
print(ohe.feature_names_in_)
# 获取每个输出特征的名称，即编码之后的特征名称。
print(ohe.get_feature_names_out())
lr = LogisticRegression()
score = cross_val_score(lr, X_train_oh, y_train, cv=5, scoring="f1")
print(score.mean())
```
![[Pasted image 20230606115004.png]]