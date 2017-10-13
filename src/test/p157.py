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

wineModel = LassoCV(cv=10).fit(X,Y)
#print(wineModel.alphas_)
#print(wineModel.mse_path_)

plot.figure()
plot.plot(wineModel.alphas_, wineModel.mse_path_,':')
#plot.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1),
#          label='Average MSE Across Folds', linewidth=2)
plot.axvline(wineModel.alphas_, linestyle='--', label='CV Estimate of Best alpha')
plot.semilogx()
plot.legend()
ax = plot.gca()
ax.invert_xaxis()
plot.xlabel('alpha')
plot.ylabel('Mean Square Error')
plot.axis('tight')
plot.show()
