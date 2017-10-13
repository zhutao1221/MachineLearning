#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
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

X_train, X_test, Y_train, Y_test = train_test_split(X_trans, Y, random_state=1)

vDepth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
sDepth = 1
acc_score_temp = 0
for iDepth in vDepth:
    clf = tree.DecisionTreeClassifier(max_depth=iDepth)
    clf = clf.fit(X_train, Y_train)
    
    temp = accuracy_score(Y_test, clf.predict(X_test))
    if temp > acc_score_temp:
       acc_score_temp = temp
       sDepth = iDepth

print(sDepth)  
print(acc_score_temp)

clf = tree.DecisionTreeClassifier(max_depth=sDepth)
clf = clf.fit(X_train, Y_train)

#with open('safe-loans.dot','w') as f:
#    f = tree.export_graphviz(clf, out_file=f)
    
with open("safe-loans.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = sDepth,
                              impurity = True,
                              feature_names = list(X_train),
                              class_names = ['not safe', 'safe'],
                              rounded = True,
                              filled= True )
     
check_call(['dot','-Tpng','safe-loans.dot','-o','safe-loans.png'])
img = Image.open("safe-loans.png")
draw = ImageDraw.Draw(img)
img.save('output.png')
PImage("output.png") 

trainPredictions =  clf.predict(X_train) 
testPredictions =  clf.predict(X_test)

fprTrain, tprTrain, thresholdsTrain = roc_curve(Y_train, trainPredictions)
roc_auc_train = auc(fprTrain, tprTrain)
print(roc_auc_train)
 
fprTest, tprTest, thresholdsTest = roc_curve(Y_test, testPredictions)
roc_auc_test = auc(fprTest, tprTest)
print(roc_auc_test) 

pl.clf()
pl.plot(fprTest, tprTrain, label='ROC curve (area = %0.2f)' % roc_auc_test)
pl.plot([0,1],[0,1], '-k')
pl.xlim(0.0, 1.0)
pl.ylim(0.0, 1.0)
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Test sample')
pl.legend(loc='lower right')
pl.show()