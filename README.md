# AI-Deep-Learning

# Theme
메일함 속 스팸 이메일의 분류

# Members
기계공학부 김민규 kmk9846a@naver.com    

전자공학부 장성윤 sdsdfg678@gmail.com     

의약생명과학과 한성목 inys7777@gmail.com

# Index
#### 1. Proposal
#### 2. Motivation & Goal
#### 3. Datasets
#### 4. Methodology
#### 5. Model evaluation & Analysis   
#### 6. Discussion


# 1. Proposal
Option A


# 2. Motivation & Goal
하루에도 수십 수백 통씩 밀려드는 이메일 보관함 속에서 스팸 메일과 그렇지 않은 정상 메일을 구별하기 위한 방법을 알아보는 것이 본 프로젝트의 주 목적이다. Naive Bayes Classification을 통해 임의의 메일이 스팸 메일일 확률을 예측하는 데이터셋 기반 Supervised learning 을 진행하고, 도출된 결과값을 바탕으로 모델의 정밀도를 확인해 본다.

# 3. Datasets
1. Train / Test datasets
   
   Kaggle에서 Spam Email CSV of 2007 TREC Public Spam Corpus와 Enron-Spam Dataset을 모아둔 데이터 셋을 학습 및 테스트 데이터로 활용했습니다.
   link: https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset

2. Extra validation datasets
    
    Kaggle에서 Apache SpamAssassin’s public datasets의 일부분을 추출한 데이터셋을 추가 검증용으로 활용했습니다.

    link: https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset


# 4. Methodology
우리는 텍스트들의 집합에서 스팸 여부를 판별하는 모델을 구성해야 합니다.

Naive Bayes는 'Bayes 법칙'에 기반한 분류기 혹은 학습 방법입니다. 각 특징들이 서로 확률적으로 독립하다는 가정이 들어가 분류를 비교적 쉽고 빠르게 할 수 있는 모델입니다.

그러나 단어들간의 연관성이 높은 글 (소설 내용 기반 장르 판별 등)처럼 서로 확률적으로 독립하다는 조건에 위반될 경우에는 정확성이 떨어질 우려가 있습니다.

스팸 메일은 단어들이 서로 연관성이 깊지 않고 특정 단어(광고성 멘트)들의 빈도수가 높다고 추정했습니다. 따라서 서로 확률적으로 독립하다는 가정이 적절히 들어 맞는다고 판단하고 Naive Bayes 방법을 이용하기로 결정했습니다.

- 스팸과 비 스팸 데이터의 분포를 살펴봅니다.
- Text data를 Embedding 하여 모델에 넣을 수 있도록 전처리합니다.
- 모델에 데이터를 넣고 학습후에 성능을 평가 합니다 (with test dataset, extra validation set)

# 5. Model evaluation & Analysis

```python
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, naive_bayes, ensemble, tree, svm, model_selection,\
                    preprocessing, feature_extraction, metrics, base, pipeline
from joblib import dump, load
import PIL
import pytesseract
import pandas as pd
```



+ 학습/테스트 데이터셋 준비

```python
file_path = './input/combined_data.csv'

df = pd.read_csv(file_path)
df.head()
```

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/0cbaca90-c12b-4959-a313-e8f6a085d682)


+ 추가 검증 데이터셋 준비

```python
extra_validation_set = pd.read_csv("./input/extra_validation_dataset.csv")
extra_validation_set = extra_validation_set[["label_num", "text"]]
extra_validation_set
```

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/5310f5d1-fbb9-42b5-9b7a-fcbf9c4a03df)


```python
labels = {0 : "Not Spam", 1 : "Spam"}
label_counts = df['label'].value_counts()
print(label_counts)
```

| Label | Count |
|-------|-------|
| 1     | 43910 |
| 0     | 39538 |



+ 스팸 비율 확인

```python
plt.pie(label_counts, labels = labels.values(), autopct = "%.2f%%")
plt.show()
```

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/9185e62a-fb9b-491b-9b99-20cf7b1add13)

+ 전체 데이터 개수 확인

```python
print(len (df))
```

83448

+ 배치마다 스팸 비율이 고르게 되었는지 확인

```python
total_emails = len(df)
batch_size = 10000
spam_counts = []
not_spam_counts = []
spam_percentages = []
for i in range(0, total_emails, batch_size):
    batch = df.iloc[i:i+batch_size]
    spam_count = batch[batch['label'] == 0]['label'].count() 
    spam_counts.append(spam_count)
    not_spam_count = batch[batch['label'] == 1]['label'].count()
    not_spam_counts.append(not_spam_count)
    spam_percentage = (spam_count / (spam_count + not_spam_count)) * 100
    spam_percentages.append(spam_percentage)
```
    
+ 시각화

```python
plt.figure(figsize=(10, 6))
plt.plot(range(len(spam_percentages)), spam_percentages, label='Spam percentage', marker='o')
plt.ylim(0, 100)
plt.xlabel('Batch Number')
plt.ylabel('Spam percentage')
plt.title('Spam percentage in Batches of 10000 Emails')
plt.legend()
plt.grid(True)
plt.show()
```

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/fb3cb65f-b0a2-4e32-8c8f-c70c8c398718)


+ 단어들의 출현 빈도 기반으로 임베딩

```python
vectorizer = feature_extraction.text.CountVectorizer()
X = df.drop('label', axis = 1).values
y = df['label'].values

X = vectorizer.fit_transform(X.reshape(-1))
print("Total number of features :", len(vectorizer.get_feature_names_out()))
print(X.shape)
```

Total number of features : 310813   
(83448, 310813)

```python
extra_X = extra_validation_set.drop('label_num', axis = 1).values
extra_y = extra_validation_set['label_num'].values

extra_X = vectorizer.transform(extra_X.reshape(-1))
```

+ 훈련 데이터, 테스트 데이터 분리   

```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.15, stratify = y)

print(X_train.shape, X_test.shape)
```

(70930, 310813) (12518, 310813)

+ 학습, 테스트 데이터의 데이터 분포도 확인

```python
plt.figure(figsize = (10, 3))

plt.subplot(1, 3, 1)
plt.title("Training Set")
plt.pie(pd.Series(y_test).value_counts(), labels = labels.values(), autopct = "%.2f%%")

plt.subplot(1, 3, 3)
plt.title("Testing Set")
plt.pie(pd.Series(y_test).value_counts(), labels = labels.values(), autopct = "%.2f%%")
plt.show()
```

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/e68e6717-bba5-46bb-adb9-0e94abf07521)


+ Naive Bayes 기반 머신러닝 모델 작성 및 학습
  
```python
model = naive_bayes.MultinomialNB()
model.fit(X_train, y_train)
```

Cross Validation Scores for model MultinomialNB() :   
[0.97730156 0.97504582 0.97596222 0.97596222 0.97497533] 0.9758494290145213

+ 테스트 데이터로 모델 예측 및 정확도 산출

```python
y_test_pred = model.predict(X_test)
accuracy = np.sum(y_test == y_test_pred) / len(y_test)
print("Accuracy (Test) : ", accuracy)
```

**Accuracy (Test) :  0.9765138200990574**

+ 테스트 데이터로 모델 성능 평가 1 (confusion matrix)

```python
cm = metrics.confusion_matrix(y_test, y_test_pred)
print(cm)

sns.heatmap(cm, annot = True)
plt.show()
```

[[5842   89]   
 [ 205 6382]]

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/c2f99f4b-404b-4a73-87e1-5a70f78308df)


+ 테스트 데이터로 모델 성능 평가 2 (confusion matrix)

```python
clf_report = metrics.classification_report(y_test, y_test_pred)
print(clf_report)
```   

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.97      | 0.98   | 0.98     | 5931    |
| 1         | 0.99      | 0.97   | 0.98     | 6587    |
|-----------|-----------|--------|----------|---------|
| Accuracy  |           |        | 0.98     | 12518   |
|-----------|-----------|--------|----------|---------|
| Macro Avg | 0.98      | 0.98   | 0.98     | 12518   |
| Weighted Avg | 0.98    | 0.98   | 0.98     | 12518   |


+ 추가 검증 데이터로 모델 성능 평가

```python
extra_predict = model.predict(extra_X)
accuracy = np.sum(extra_y == extra_predict) / len(extra_predict)

print("Accuracy (Extra validation) : ", accuracy)
```

**Accuracy (Extra validation) :  0.9694449816283117**

```python
cm = metrics.confusion_matrix(extra_y, extra_predict)
print(cm)

sns.heatmap(cm, annot = True)
plt.show()
```

[[3575   97]    
 [  61 1438]]

![image](https://github.com/Inys7777/AI-Deep-Learning/assets/150836586/c82964eb-c938-4885-ac9d-b212e01ef60f)


+ 추가 검증 데이터로 모델 성능 평가 2 (confusion matrix)

```python

clf_report = metrics.classification_report(extra_y, extra_predict)
print(clf_report)
```

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.98      | 0.97   | 0.98     | 3672    |
| 1         | 0.94      | 0.96   | 0.95     | 1499    |
|-----------|-----------|--------|----------|---------|
| Accuracy  |           |        | 0.97     | 5171    |
|-----------|-----------|--------|----------|---------|
| Macro Avg | 0.96      | 0.97   | 0.96     | 5171    |
| Weighted Avg | 0.97    | 0.97   | 0.97     | 5171    |



# 6. Discussion
테스트 데이터, 추가 검증 데이터 모두에서 정확도, 재현율, F1-score 등의 평가 기준에서 높은 점수가 나왔습니다.
Methodology에서 예측했던 대로 Naive Bayes 모델은 Spam 메일 filtering에 높은 성능을 보였습니다.

그러나, 해당 데이터 셋이 각각, 2007년(학습, 테스트) 2003년(추가 검증) 등 10~20년전 자료이기 때문에 현대의 스팸 이메일 필터링에 쓰기에는 어려울 수 있다는 판단이 들었습니다.
조금 더 시간의 여유가 있어 현대의 스팸 이메일 데이터셋을 확보하여 학습 및 검증에 이용할 수 있었다면 좋을 것 같습니다.

만약 이 스팸 모델 분류기를 좀 더 발전 시킨다면 현대의 많은 데이터셋을 확보하고 Naive Bayes 뿐만이 아닌 여러가지 모델을 비교하여 발전 시키고 싶습니다.

# Youtube Link   
<https://youtu.be/j-poW_dLjtc>

# References
- Train/Test Dataset: https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset
- Extra validation Dataset: https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset
- Reference code: https://www.kaggle.com/code/prasadmeesala/spam-email-classification-naivebayes
