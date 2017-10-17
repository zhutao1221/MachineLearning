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

nrows = len(xList)
ncols = len(xList[0])

X = numpy.array(xList)
Y = numpy.array(labels)
wineNames = numpy.array(names)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.30,
                                                random_state=531)

nEst = 2000
depth = 7
learnRate = 0.01
subSamp =0.5
wineGBMModel = ensemble.GradientBoostingRegressor(n_estimators=nEst, 
                                                  max_depth=depth,
                                                  learning_rate=learnRate,
                                                  subsample=subSamp,
                                                  loss='ls')

wineGBMModel.fit(xTrain, yTrain)

msError = []
predictions = wineGBMModel.staged_predict(xTest)
for p in predictions:
    msError.append(mean_squared_error(yTest, p))
    
print('MSE')
print(min(msError))

plot.figure()
plot.plot(range(1, nEst + 1), wineGBMModel.train_score_,
          label='Training Set MSE')
plot.plot(range(1, nEst + 1), msError, label='Test Set MSE')
plot.legend(loc='upper right')
plot.xlabel('Number of Trees in Ensembles')
plot.ylabel('Mean Squared Error')
plot.show()

featureImportance = wineGBMModel.feature_importances_
featureImportance = featureImportance / featureImportance.max()

idxSorted = numpy.argsort(featureImportance)
barPos = numpy.arange(idxSorted.shape[0]) + .5
plot.barh(barPos, featureImportance[idxSorted], align='center')
plot.yticks(barPos, wineNames[idxSorted])
plot.xlabel('Variable Importance')
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()
