#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import csv
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

xList = []
labels = []
names = []
firstLine = True

with open('winequality-red.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        if (firstLine):
            names = row
            firstLine = False
        else:
            labels.append(float(row[-1]))
            row.pop()
            floatRow = [float(num) for num in row]
            xList.append(floatRow)
print(names)
nrows = len(xList)
ncols = len(xList[0])

X = numpy.array(xList)
Y = numpy.array(labels)
wineNames = numpy.array(names)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30,random_state=531)

mseOos = []
nTreesList = range(50, 500, 10)
for iTrees in nTreesList:
    depth = None
    maxFeat = 4
    wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees,
                                                 max_depth=depth, max_features=maxFeat,
                                                 oob_score=False, random_state=531)
    wineRFModel.fit(xTrain,yTrain)
    
    prediction = wineRFModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))
    
print('MSE')
print(mseOos[-1])

plot.plot(nTreesList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
plot.show() 

featureImportance = wineRFModel.feature_importances_

featureImportance = featureImportance / featureImportance.max()

sorted_idx = numpy.argsort(featureImportance)
barPos = numpy.arange(sorted_idx.shape[0]) + .5
plot.barh(barPos, featureImportance[sorted_idx], align='center')
plot.yticks(barPos, wineNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.show()