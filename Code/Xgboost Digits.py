# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:38:22 2021

@author: su
"""

import xgboost
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
digits = load_digits()
X = digits.data
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 5,                 # the maximum depth of each tree
    'eta': 0.3,                     # the training step for each iteration
    'silent': 1,                    # logging mode - quiet
    'objective': 'multi:softmax',   # multiclass classification using the softmax objective
    'num_class': 10                 # the number of classes that exist in this datset
}  
num_round = 500  # the number of training iterations



bstmodel = xgboost.train(param, dtrain, num_round)

#Save as human readable model
bstmodel.dump_model('dump.raw.txt')

preds = bstmodel.predict(dtest)
#print(preds)
for i in range(20):
    print(y_test[i])

from sklearn import metrics
acc = metrics.accuracy_score(y_test, preds)
print('Accuracy: %.2f%%' % (acc*100.0))