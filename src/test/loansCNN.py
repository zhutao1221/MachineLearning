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
    
     
    
    #xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.70,random_state=531)
    #yTrain.dtype = 'int64'
    #print(yTrain)
    return X,Y  

def linerRegression(train_x,train_y,epoch=90000,rate = 0.0001): 
    
    with tf.variable_scope('input'): 
        n = train_x.shape[0]  
        x = tf.placeholder(tf.float32,name='x-input')  
        y = tf.placeholder(tf.float32,name='y-input')
    
    with tf.variable_scope('layer1_fc1'):      
        fc1_weights = tf.get_variable(
            'weight1', [11, 10],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        fc1_biases = tf.get_variable(
            'bias1', [10], initializer=tf.constant_initializer(0.1)) 
        fc1 = tf.nn.relu(tf.add(tf.matmul(x,fc1_weights),fc1_biases))
        
    with tf.variable_scope('layer2_fc2'):      
        fc2_weights = tf.get_variable(
            'weight2', [10, 5],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        fc2_biases = tf.get_variable(
            'bias2', [5], initializer=tf.constant_initializer(0.1)) 
        fc2 = tf.nn.relu(tf.add(tf.matmul(fc1,fc2_weights),fc2_biases))
        
    with tf.variable_scope('output'):      
        fc3_weights = tf.get_variable(
            'weight3', [5, 2],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        fc3_biases = tf.get_variable(
            'bias3', [2], initializer=tf.constant_initializer(0.1)) 
        output = tf.add(tf.matmul(fc2,fc3_weights),fc3_biases)                  
        
    with tf.name_scope('loss_function'):          
        #loss = tf.reduce_sum(tf.square(output-y))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = train_y, logits = output)
    
        loss = tf.reduce_mean(cross_entropy)        
    
    with tf.name_scope('train_step'):      
        optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)  
  
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()     
    with tf.Session() as sess:
        tf.global_variables_initializer().run()     
        for index in range(epoch):   
            sess.run(optimizer,{x:train_x,y:train_y})
            print(index)
        #print ('output is ',sess.run(output,{x:train_x,y:train_y}))  
        print ('loss is ',sess.run(loss,{x:train_x,y:train_y}))
        
        saver.save(sess, 'model/model.ckpt')
        outTemp = sess.run(output,{x:train_x,y:train_y})
        
#         out = tf.argmax(outTemp,1)
#         print(out)  
        
        
        #print ('output is ',sess.run(output,{x:train_x,y:train_y}))    
        #writer = tf.summary.FileWriter("/tmp/tflog", tf.get_default_graph())
        #writer.close()    
   
    return outTemp

if __name__ == "__main__":  
    train_x,train_y = createData()
     
    out = linerRegression(train_x,train_y)
    print(out)
    
    i=0
    count=0
    tempxx=0
    for i in range(train_y.shape[0]):
        if out[i][0] > out[i][1]:
            tempxx=0
        else:  
            tempxx=1   
        if train_y[i] == tempxx:
            count = count + 1
        i = i + 1
    print(count)  
 