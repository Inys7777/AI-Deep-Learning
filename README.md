# AI-Deep-Learning

# Theme
메일함 속 스팸 이메일의 분류

# Members
기계공학부 김민규 kmk9846a@naver.com    

전자공학부 장성윤 sdsdfg678@gmail.com     

의약생명과학과 한성목 inys7777@gmail.com

# Proposal
Option A

# Motivation & Goal
하루에도 수십 수백 통씩 밀려드는 이메일 보관함 속에서 스팸 메일과 그렇지 않은 정상 메일을 구별하기 위한 방법을 알아보는 것이 본 프로젝트의 주 목적이다. Naive Bayes Classification을 통해 임의의 메일이 스팸 메일일 확률을 예측하는 데이터셋 기반 Supervised learning 을 진행하고, 도출된 결과값을 바탕으로 모델의 정밀도를 확인해 본다.

# Datasets
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


```python
# 학습/테스트 데이터셋 준비
file_path = './input/combined_data.csv'

df = pd.read_csv(file_path)
df.head()

```python
# 추가 검증 데이터셋 준비
extra_validation_set = pd.read_csv("./input/extra_validation_dataset.csv")
extra_validation_set = extra_validation_set[["label_num", "text"]]
extra_validation_set

```python
labels = {0 : "Not Spam", 1 : "Spam"}
label_counts = df['label'].value_counts()
print(label_counts)



