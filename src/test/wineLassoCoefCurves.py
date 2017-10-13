#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import csv
from sklearn import  datasets, linear_model
from sklearn.linear_model import LassoCV
from math import sqrt
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

xMeans = []
xSD = []
for i in range(ncols):
    col = [xList[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    xMeans.append(mean)
    colDiff = [(xList[j][i] - mean) for j in range(nrows)]
    sumSq = sum(colDiff[i]**2 for i in range(nrows))
    stdDev = sqrt(sumSq/nrows)
    xSD.append(stdDev)
    
xNormalized = []
for i in range(nrows):
    rowNormalizd = [(xList[i][j] - xMeans[j])/xSD[j] \
                    for j in range(ncols)]
    xNormalized.append(rowNormalizd)
    
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([((labels[i] - meanLabel)**2) for i in range(nrows)])/nrows)
labelNormalized = [(labels[i] - meanLabel) / sdLabel \
                   for i in range(nrows)]


Y = numpy.array(labelNormalized)
X = numpy.array(xNormalized)

alphas, coefs, _=linear_model.lasso_path(X, Y, return_models=False)

plot.plot(alphas, coefs.T)
plot.xlabel('alpha')
plot.ylabel('Coefficients')
plot.axis('tight')
plot.semilogx()
ax = plot.gca()
ax.invert_xaxis()
plot.show()

nattr, nalpha = coefs.shape
nzList = []
for iAlpha in range(1,nalpha):
    coefList = list(coefs[:,iAlpha])
    nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
    for q in nzCoef:
        if not(q in nzList):
            nzList.append(q)