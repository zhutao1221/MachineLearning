#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plot
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from math import floor
import random

nPoints = 1000
xPlot = [(float(i) / float(nPoints) - 0.5) for i in range(nPoints + 1)]

x = [[s] for s in xPlot]
numpy.random.seed(1)
y = [s + numpy.random.normal(scale = 0.1) for s in xPlot]

nSample = int(nPoints * 0.30)
idxTest = random.sample(range(nPoints), nSample)
idxTest.sort()
idxTrain = [idx for idx in range(nPoints) if not(idx in idxTest)]

xTrain = [x[r] for r in idxTrain]
xTest = [x[r] for r in idxTest]
yTrain = [y[r] for r in idxTrain]
yTest = [y[r] for r in idxTest]

numTreesMax = 30

treeDepth = 5

modelList = []
predList = []
eps = 0.1
residuals = list(yTrain)

for iTrees in range(numTreesMax):
    
    modelList.append(DecisionTreeRegressor(max_depth=treeDepth))
    modelList[-1].fit(xTrain, residuals)
    
    latestInSamplePrediction = modelList[-1].predict(xTrain)
    
    residuals = [residuals[i] - eps * latestInSamplePrediction[i] for i in range(len(residuals))]
    
    latestOutSamplePrediction = modelList[-1].predict(xTest)
    predList.append(list(latestOutSamplePrediction))
    
mse = []
allPredictions = []
for iModels in range(len(modelList)):
    prediction = []
    for iPred in range(len(xTest)):
        prediction.append(sum([predList[i][iPred]
        for i in range(iModels + 1)]) * eps)
    
    allPredictions.append(prediction)
    errors = [(yTest[i] - prediction[i]) for i in range(len(yTest))]
    mse.append(sum([e**2 for e in errors]) / len(yTest))
    
nModels = [i + 1 for i in range(len(modelList))]

plot.plot(nModels,mse)
plot.axis('tight')
plot.xlabel('Number of Models in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0,max(mse)))
plot.show()

    
plotList = [0, 14, 29]
lineType = [':', '-.', '--']
plot.figure()
for i in range(len(plotList)):
    iPlot = plotList[i]
    textLegend = 'Prediction with ' + str(iPlot) +  ' Trees'
    plot.plot(xTest, allPredictions[iPlot], label = textLegend,
              linestyle = lineType[i])
plot.plot(xTest, yTest, label='True y Value', alpha=0.25)
plot.legend(bbox_to_anchor=(1,0.3))
plot.axis('tight')
plot.xlabel('x value')
plot.ylabel('Predictions')
plot.show()    