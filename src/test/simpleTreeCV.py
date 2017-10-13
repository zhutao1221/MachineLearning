#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plot

nPoints = 100

xPlot = [(float(i)/float(nPoints) - 0.5) for i in range(nPoints + 1)]
#print(len(xPlot))

x = [[s] for s in xPlot]
numpy.random.seed(1)
y = [s + numpy.random.normal(scale=0.1) for s in xPlot]

nrow = len(x)

depthList = [1,2,3,4,5,6,7]
xvalMSE = []
nxval = 10

for iDepth in depthList:
    for ixval in range(nxval):
        print(ixval)
        ideTest = [a for a in range(nrow) if a%nxval == ixval%nxval]
        idxTrain = [a for a in range(nrow) if a%nxval != ixval%nxval]
        
print(ideTest)    
print(idxTrain)     