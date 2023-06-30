有些单词，我们不能仅以当前文档中的频数来进行衡量，还要考虑其在语料库中，在其他文档 中出现的次数。因为有些单词，确实是非常常见的，其在语料库所有的文档中，可能都会频繁出现，对于这样的单 词，我们就应该降低其重要性。例如，在新闻联播中，“中国”，“发展”等单词，在语料库中出现的频率非常高，因此，即使这些词在某篇文档中频繁出现，也不能说明这些词对当前文档是非常重要的，因此这些词并不含有特别有意义的信息。 
TF-IDF可以用来调整单词在文档中的权重。其由两部分组成：
1. TF（Term-Frequency）词频，指一个单词在文档中出现的频率。 
2. IDF（Inverse Document-Frequency）逆文档频率。
计算方式为：$$idf\left(t\right)=\log\frac n{1+df\left(t\right)}$$
$$tf-idf(t,d)=tf(t,d)*idf(t)$$
+ $t$：某单词
+ $n$：语料库中文档的总数
+ $df(t)$：语料库中含有单词t的文档个数
说明：scikit-learn库中实现的tf-idf转换，与标准的公式略有不同。并且，tf-idf结果会使用L1或L2范数进行标准化 （规范化）处理。

``` python
from sklearn.feature_extraction.text import TfidfTransformer 
count = CountVectorizer() 
docs = [ 
		"Where there is a will, there is a way.", "There is no royal road to learning.", 
		] 
bag = count.fit_transform(docs) 
tfidf = TfidfTransformer() 
t = tfidf.fit_transform(bag) 
# TfidfTransformer转换的结果也是稀疏矩阵。 
print(t.toarray())
```
![[Pasted image 20230530202124.png]]
此外，scikit-learn中，同时提供了一个类TfidfVectorizer，其可以直接将文档转换为TF-IDF值，也就是说，该类相 当于集成了CountVectorizer与TfidfTransformer两个类的功能，这对我们实现上提供了便利。

``` python
from sklearn.feature_extraction.text import TfidfVectorizer 
docs = [ "Where there is a will, there is a way.", "There is no royal road to learning.", ] 
tfidf = TfidfVectorizer() 
t = tfidf.fit_transform(docs) 
print(t.toarray())
```
![[Pasted image 20230530202225.png]]
可以看出，这与我们之前使用CountVectorizer与TfidfTransformer两个类转换的结果是一样的。