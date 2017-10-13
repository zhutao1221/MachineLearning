#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import csv
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
from math import sqrt
import random
import matplotlib.pyplot as plot

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

random.seed(1)
nSample = int(nrows * 0.30)
idxTest = random.sample(range(nrows),nSample)
idxTest.sort()
idxTrain = [idx for idx in range(nrows) if not (idx in idxTest)]

xTrain = [xList[r] for r in idxTrain]
xTest = [xList[r] for r in idxTest]
yTrain = [labels[r] for r in idxTrain]
yTest = [labels[r] for r in idxTest]

numTreesMax = 30

treeDepth = 100

modelList = []
predList = []

nBagSamples = int(len(xTrain) * 0.5)

for iTrees in range(numTreesMax):
    idxBag = []
    for i in range(nBagSamples):
        idxBag.append(random.choice(range(len(xTrain))))
    xTrainBag = [xTrain[i] for i in idxBag]
    yTrainBag = [yTrain[i] for i in idxBag]
    
    modelList.append(DecisionTreeRegressor(max_depth=treeDepth))
    modelList[-1].fit(xTrainBag, yTrainBag)
    
    latestPrediction = modelList[-1].predict(xTest)
    predList.append(list(latestPrediction))
    
mse = []
allPredictions = []

for iModels in range(len(modelList)):
    
    prediction = []
    for iPred in range(len(xTest)):
        prediction.append(sum([predList[i][iPred] 
        for i in range(iModels + 1)]) / (iModels + 1))
    
    allPredictions.append(prediction)
    errors = [(yTest[i] - prediction[i]) for i in range(len(yTest))]
    mse.append(sum([e**2 for e in errors]) / len(yTest))
    
nModels = [i + 1 for i in range(len(modelList))]
plot.plot(nModels,mse)
plot.axis('tight')
plot.xlabel('Number of Tree Models in Ensembel')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0, max(mse)))
plot.show() 