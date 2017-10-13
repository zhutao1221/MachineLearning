import numpy
import csv
from sklearn import  datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

def xattrSelect(x, idxSet):
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return(xOut)

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

attributeList = []
index = range(len(xList[1]))
indexSet = set(index)
indexSeq = []
oosError = []

for i in index:
    attSet = set(attributeList)
    attTrySet = indexSet - attSet
    #print(indexSet)
    #print(attSet)    
    attTry = [ii for ii in attTrySet]
    errorList = []
    attTemp = []
    
    for iTry in attTry:
       attTemp = [] + attributeList
       attTemp.append(iTry)
       
       xTrainTemp = xattrSelect(xListTrain, attTemp)
       xTestTemp = xattrSelect(xListTest, attTemp)
       
       xTrain = numpy.array(xTrainTemp)
       yTrain = numpy.array(labelTrain)
       xTest = numpy.array(xTestTemp)
       yTest = numpy.array(labelTest)
       
       wineQModel = linear_model.LinearRegression()
       wineQModel.fit(xTrain, yTrain)
       yTest = yTest.astype('float64')
       rmsError = numpy.linalg.norm((yTest-wineQModel.predict(xTest)),2)/sqrt(len(yTest))
       errorList.append(rmsError)
       attTemp = []
          
    iBest = numpy.argmin(errorList)
    attributeList.append(attTry[iBest])
    oosError.append(errorList[iBest])
print("Out of sample error versus attribute set size")
print(oosError)
print("\n" + "Best attribute indices")
print(attributeList)

namesList = [ names[i] for i in attributeList]
print("\n" + "Best attribute names")
print(namesList)

x = range(len(oosError))
plt.plot(x, oosError,'k')
plt.xlabel('Number of Attributes')
plt.ylabel('Error (RMS)')
plt.show()

indexBest = oosError.index(min(oosError))
attributesBest = attributeList[0:(indexBest+1)]

xTrainTemp = xattrSelect(xListTrain, attributesBest)
xTestTemp = xattrSelect(xListTest, attributesBest)
xTrain = numpy.array(xTrainTemp);xTest = numpy.array(xTestTemp)

wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain,yTrain)
errorVector = yTest - wineQModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

plt.scatter(wineQModel.predict(xTest),yTest,s=100,alpha=0.10)
plt.xlabel('Predicted Taste Score')
plt.ylabel('Actual Taste Score')
plt.show()