import numpy
import random
from sklearn import datasets,linear_model
from sklearn.metrics import roc_curve,auc
import pylab as pl
import csv

def confusionMatrix(predicted,actual,threshold):
    if len(predicted) != len(actual): return -1
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    rtn = [tp,fn,fp,tn]
    return rtn

xList = []
lables = []

with open('sonar.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        if (row[-1] == 'M'):
            lables.append(1.0)
        else:
            lables.append(0.0)
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

#print(xTrain.shape)
#print(yTrain.shape)
#print(xTest.shape)
#print(yTest.shape) 

rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain, yTrain)

trainingPredictions = rocksVMinesModel.predict(xTrain)

print("some values predicted by model",trainingPredictions[0:5],trainingPredictions[-6:-1])

confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)
tp = confusionMatTrain[0];fn = confusionMatTrain[1]
fp = confusionMatTrain[2];tn = confusionMatTrain[3]
print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn) + "\n")

testPredictions = rocksVMinesModel.predict(xTest)

conMatTest = confusionMatrix(testPredictions, yTest, 0.5)
tp = conMatTest[0];fn = conMatTest[1]
fp = conMatTest[2];tn = conMatTest[3]
print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn) + "\n")

fpr,tpr,thresholds = roc_curve(yTrain,trainingPredictions)
roc_auc = auc(fpr, tpr)
print('AUC for in-sample ROC curve: %f' % roc_auc)

pl.clf()
pl.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0,1], [0,1], 'k-')
pl.xlim([0.0,1.0])
pl.ylim([0.0,1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('In sample ROC rocks versus mines')
pl.legend(loc="lower right") 
pl.show()

fpr,tpr,thresholds = roc_curve(yTest,testPredictions)
roc_auc = auc(fpr, tpr)
print('AUC for out-of-sample ROC curve: %f' % roc_auc)

pl.clf()
pl.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0,1], [0,1], 'k-')
pl.xlim([0.0,1.0])
pl.ylim([0.0,1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Out of sample ROC rocks versus mines')
pl.legend(loc="lower right") 
pl.show()