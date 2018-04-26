import os
import tensorflow as tf
import sonar_inference,sonar_train
import numpy as np
import csv

MODEL_SAVE_PATH = 'model'
MODEL_NAME = 'model.ckpt'

def createData():
    xList = []
    labels = []

    with open('data/sonar.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[-1]))
            row.pop()
            floatRow = [float(num) for num in row]
            xList.append(floatRow)

    X = np.array(xList)
    Y = np.array(labels)    
    return X,Y  

def evaluate(testX,testY):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,name='x-input')
        