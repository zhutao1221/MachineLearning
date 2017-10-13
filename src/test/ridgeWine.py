import numpy
import csv
from sklearn import  datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

xList = []
lables = []
names = []
firstLine = True

with open('winequality-red.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        if (firstLine):
            names = row
            firstLine = False
        else:
            lables.append(row[-1])
            row.pop()
            floatRow = [float(num) for num in row]
            xList.append(floatRow)
            
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelTest = [lables[i] for i in indices if i%3 == 0]
labelTrain = [lables[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain);yTrain = numpy.array(labelTrain)
xTest = numpy.array(xListTest);yTest = numpy.array(labelTest)

alphaList = [0.1**i for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

rmsError = []
for alph in alphaList:
    wineRidgeModel = linear_model.Ridge(alpha=alph)
    wineRidgeModel.fit(xTrain,yTrain)
    yTest = yTest.astype('float64')
    rmsError.append(numpy.linalg.norm((yTest - wineRidgeModel.predict(xTest)),2)/sqrt(len(yTest)))
print("RMS Error   alpha")
for i in range(len(rmsError)):
    print(rmsError[i],alphaList[i])
    
x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.show()

indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
wineRidgeModel = linear_model.Ridge(alpha=alph)
wineRidgeModel.fit(xTrain, yTrain)
errorVector = yTest-wineRidgeModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel('Bin Boundaries')
plt.ylabel('Counts')
plt.show()

plt.scatter(wineRidgeModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted Taste Score')
plt.ylabel('Actual Taste Score')
plt.show()