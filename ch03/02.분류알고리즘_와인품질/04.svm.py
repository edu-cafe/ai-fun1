# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:18:00 2018

@author: kim
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:44:32 2018
http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
@author: kim
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import *

#os.chdir('c:/share/lab')
df = pd.read_csv('./preprocessed_redwine_quality.csv', engine='python', encoding='cp949')

train, test = train_test_split(df, test_size=0.3)    #train:test 비율을 7:3으로 자름

train_x = train.drop('품질수준', 1)    # 1: col
train_y = train['품질수준']
test_x = test.drop('품질수준', 1)
test_y = test['품질수준']

kernel_list = ['linear', 'rbf', 'sigmoid']
C_list = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]

for kernel in kernel_list:
    for C in C_list:
        model = svm.SVC(C=C, kernel=kernel).fit(train_x, train_y)
        predited = model.predict(test_x)
        print(kernel, C, accuracy_score(test_y, predited))
        

























