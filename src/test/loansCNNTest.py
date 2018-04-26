#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf  
import numpy as np
import csv
from sklearn.cross_validation import train_test_split  

def createData():
    xList = []
    labels = []
    names = []
    firstLine = True

    with open('loansTemp.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            if (firstLine):
                names = row
                firstLine = False
            else:
                labels.append(int(row[-1]))
                row.pop()
                floatRow = [float(num) for num in row]
                xList.append(floatRow)
    nrows = len(xList)
    ncols = len(xList[0])

    X = np.array(xList)
    Y = np.array(labels)
    
     
    
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.70,random_state=531)
    #yTrain.dtype = 'int64'
    #print(yTrain)
    return xTrain,yTrain  

def linerRegression(train_x):

    saver = tf.train.Saver() 
        
    with tf.Session() as sess:
        saver.restore(sess,'model/model.ckpt')

        print(sess.run(train_x))
#         out = tf.argmax(outTemp,1)
#         print(out)  
        
        
        #print ('output is ',sess.run(output,{x:train_x,y:train_y}))    
        #writer = tf.summary.FileWriter("/tmp/tflog", tf.get_default_graph())
        #writer.close()    
   
    return

if __name__ == "__main__":  
    train_x,train_y = createData()
     
    out = linerRegression(train_x)
    
    
#     i=0
#     count=0
#     for i in range(train_y.shape[0]):
#         if train_y[i] == out[i]:
#             count = count + 1
#         i = i + 1
#     print(count)  
 