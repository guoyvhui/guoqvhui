
---

# 文本分类项目：邮件分类

## 项目简介
在本项目中，我们采用多项式朴素贝叶斯分类器对邮件进行分类，并提供了两种特征提取方式以供选择：一种是基于高频词的特征选择，另一种是运用TF-IDF进行特征加权。通过改变特征提取模式，能够直观地对比不同特征提取方法对分类效果产生的影响。

## 特征模式介绍

### 1. 高频词特征选择

#### 原理
高频词特征选择是依据词频（Term Frequency, TF）来进行的。它通过计算每个词在文档中出现的频次，从中挑选出那些出现次数最多的词汇作为特征。该方法的基本假设是，那些频繁出现的词能够较好地指示文档所属的类别。

#### 数学表达形式
假设文档集合为 D，其中包含 N 个文档 d 
1
​
 ,d 
2
​
 ,…,d 
N
​
 ，每个文档 d 
i
​
  包含一系列词 w 
1
​
 ,w 
2
​
 ,…,w 
M
​
 。词频表示词 w 在文档 d 中出现的次数。高频词特征选择的步骤如下：
统计所有文档中每个词的总词频：将每个词在所有文档中出现的次数相加，得到该词的总词频。
2. 按照总词频从高到低排序，选择前 \( K \) 个词作为特征。

#### 实现逻辑
1. 遍历所有文档，统计每个词的出现次数。
2. 对词频进行排序，选择前 \( K \) 个高频词作为特征。
3. 构建特征向量时，检查文档中是否包含这些高频词，包含则标记为1，否则标记为0。

#### 优点
- 实现简单，计算效率高。
- 对于一些简单的文本分类任务，高频词可以提供有效的特征。

#### 缺点
- 忽略了词的区分能力，高频词可能并不总是对分类有帮助。
- 无法处理词的权重问题，所有特征词的权重相同。

### 2. TF-IDF特征加权

#### 原理
TF-IDF（词频-逆文档频率）是一种用于评估词语重要性的统计方法。它结合了词频（TF）和逆文档频率（IDF）两个因素，来确定词语在文档中的重要程度。一个词语的TF-IDF值越高，表明该词语对于文档的分类越具有关键性。

#### 数学表达形式
- **词频（TF）**：词 \( w \) 在文档 \( d \) 中出现的次数。
  $\[
  \text{TF}(w, d) = \frac{\text{词 } w \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}
  \]$
- **逆文档频率（IDF）**：衡量词的普遍重要性。
  $\[
  \text{IDF}(w) = \log \frac{\text{文档总数}}{\text{包含词 } w \text{ 的文档数}}
  \]$
- **TF-IDF值**：
  $\[
  \text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)
  \]$

#### 实现逻辑
1. 计算每个词的TF值。
2. 计算每个词的IDF值。
3. 通过TF-IDF公式计算每个词的权重。
4. 构建特征向量时，将每个词的TF-IDF值作为特征值。

#### 优点
- 考虑了词的区分能力，能够更好地衡量词的重要性。
- 对于文本分类等任务，TF-IDF特征通常比高频词特征更有效。

#### 缺点
- 计算复杂度较高，需要计算TF和IDF值。
- 对于一些非常稀疏的词，TF-IDF值可能过高，需要进行平滑处理。

## 高频词/TF-IDF两种特征模式的切换方法

在代码中，可以通过设置一个参数（如 `feature_mode`）来切换两种特征模式。以下是具体的实现方法：

### 示例代码

#### 数据预处理
```python
import os
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# 数据预处理
def preprocess_text(text):
    # 分词
    words = jieba.cut(text)
    # 停用词过滤
    stop_words = set(['的', '是', '和', ...])  # 替换为实际的停用词列表
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
```

#### 高频词特征选择
```python
# 高频词特征选择
def select_high_freq_words(documents, top_n=1000):
    all_words = []
    for doc in documents:
        words = doc.split()
        all_words.extend(words)
    word_counts = Counter(all_words)
    high_freq_words = [word for word, count in word_counts.most_common(top_n)]
    return high_freq_words
```

#### TF-IDF特征加权
```python
# TF-IDF特征加权
def calculate_tf_idf(documents):
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(documents)
    return tf_idf_matrix, vectorizer.get_feature_names_out()
```

#### 主函数
```python
# 主函数
def main():
    # 读取邮件数据
    documents = []
    labels = []
    for filename in os.listdir('邮件_files'):
        with open(os.path.join('邮件_files', filename), 'r', encoding='utf-8') as fr:
            text = fr.read()
            documents.append(preprocess_text(text))
            # 假设文件名中包含标签信息，如'0.txt'表示非垃圾邮件，'1.txt'表示垃圾邮件
            labels.append(0 if filename.startswith('0') else 1)

    # 特征模式切换
    feature_mode = 'high_freq'  # 高频词特征模式
    # feature_mode = 'tf_idf'  # TF-IDF特征模式

    if feature_mode == 'high_freq':
        # 高频词特征选择
        features = select_high_freq_words(documents, top_n=1000)
        # 构建特征向量
        feature_vectors = []
        for doc in documents:
            words = doc.split()
            vector = [1 if word in words else 0 for word in features]
            feature_vectors.append(vector)
    elif feature_mode == 'tf_idf':
        # TF-IDF特征加权
        tf_idf_matrix, feature_names = calculate_tf_idf(documents)
        feature_vectors = tf_idf_matrix.toarray()

    # 训练模型
    # ...

if __name__ == '__main__':
    main()
```

### 切换方法
1. **高频词特征模式**：
   - 设置 `feature_mode = 'high_freq'`。
   - 调用 `select_high_freq_words` 函数选择高频词作为特征。
   - 构建特征向量时，检查文档中是否包含这些高频词，包含则标记为1，否则标记为0。

2. **TF-IDF特征模式**：
   - 设置 `feature_mode = 'tf_idf'`。
   - 调用 `calculate_tf_idf` 函数计算TF-IDF值。
   - 构建特征向量时，将每个词的TF-IDF值作为特征值。

通过设置 `feature_mode` 的值，可以在两种特征模式之间轻松切换，从而比较不同特征提取方法对模型性能的影响。

---
<img src="https://github.com/guoyvhui/guoqvhui/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-06-19%20184122.png" width="200" alt="截图">