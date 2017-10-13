#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from collections import defaultdict
from sklearn import tree
from subprocess import check_call
from IPython.display import Image as PImage
from PIL import Image, ImageDraw, ImageFont
import pylab as pl

df = pd.read_csv('loans.csv')

X = df.drop('safe_loans', axis=1)
Y = df.safe_loans

d = defaultdict(LabelEncoder)
X_trans = X.apply(lambda x: d[x.name].fit_transform(x))
X_trans_np = np.array(X_trans)
Y_trans_np = np.array(Y)

nrows = len(X_trans_np)
ncols = len(X_trans_np[0])

xMeans = []
xSD = []
i = 0 
for i in range(ncols):
    col = [X_trans_np[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    xMeans.append(mean)
    colDiff = [(X_trans_np[j][i] - mean) for j in range(nrows)]
    sumSq = sum(colDiff[i]**2 for i in range(nrows))
    stdDev = sqrt(sumSq/nrows)
    xSD.append(stdDev)
    
xNormalized = []
i = 0
for i in range(nrows):
    rowNormalizd = [(X_trans_np[i][j] - xMeans[j])/xSD[j] \
                    for j in range(ncols)]
    xNormalized.append(rowNormalizd)
    
meanLabel = sum(Y_trans_np)/nrows
sdLabel = sqrt(sum([((Y_trans_np[i] - meanLabel)**2) for i in range(nrows)])/nrows)
labelNormalized = [(Y_trans_np[i] - meanLabel) / sdLabel \
                   for i in range(nrows)]    

#print(len(xNormalized))
#print(len(xNormalized[0]))
#print(len(Y_trans_np))

X_train, X_test, Y_train, Y_test = train_test_split(xNormalized, labelNormalized, random_state=1)
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, Y_train)
#vDepth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#sDepth = 1
#acc_score_temp = 0
#for iDepth in vDepth:
#   clf = tree.DecisionTreeClassifier(max_depth=iDepth)
#    clf = clf.fit(X_train, Y_train)
    
#    temp = accuracy_score(Y_test, clf.predict(X_test))
#    if temp > acc_score_temp:
#       acc_score_temp = temp
#       sDepth = iDepth

#print(sDepth)  
#print(acc_score_temp)

#clf = tree.DecisionTreeClassifier(max_depth=sDepth)
#clf = clf.fit(X_train, Y_train)