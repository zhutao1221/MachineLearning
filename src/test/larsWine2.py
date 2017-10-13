import numpy
import csv
from sklearn import  datasets, linear_model
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
beta = [0.0] * ncols
betaMat = []
betaMat.append(list(beta))

nSteps = 350
stepSize = 0.004

for i in range(nSteps):
    residuals = [0.0] * nrows
    for j in range(nrows):
        labelsHat = sum([xNormalized[j][k] * beta[k] for k in range(ncols)])
        residuals[j] = labelNormalized[j] - labelsHat

    corr = [0.0] * ncols

    for j in range(ncols):
        corr[j] = sum([xNormalized[k][j] * residuals[k] \
                       for k in range(nrows)]) / nrows
                       
    iStar = 0
    corrStar = corr[0]
    
    for j in range(1, (ncols)):
        if abs(corrStar) < abs(corr[j]):
            iStar =j;corrStar = corr[j]

    beta[iStar] += stepSize * corrStar / abs(corrStar)
    betaMat.append(list(beta))
    
for i in range(ncols):
    coefCurve = [betaMat[k][i] for  k in range(nSteps)]
    xaxis = range(nSteps)
    plot.plot(xaxis, coefCurve)
       
plot.xlabel('Step Taken')
plot.ylabel('CoefCurve Values')
plot.show()                                          
   