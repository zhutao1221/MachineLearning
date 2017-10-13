#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import csv
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
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

wineTree = DecisionTreeRegressor(max_depth=3)

wineTree.fit(xList, labels)

with open('wineTree.dot','w') as f:
    f = tree.export_graphviz(wineTree, out_file=f)