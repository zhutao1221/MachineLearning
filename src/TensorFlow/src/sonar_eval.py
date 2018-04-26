import os
import tensorflow as tf
import sonar_inference,sonar_train
import numpy as np
import csv

EVAL_INTERVAL_SECS = 10

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
        y_ = tf.placeholder(tf.float32,name='y-input')
                
        y = sonar_inference.inference(x, 0, None)
        
        variable_averages = tf.train.ExponentialMovingAverage(sonar_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('model/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                accuracy_score = sess.run(y,feed_dict={x: testX, y_:testY})
                return accuracy_score
                    
def main(argv=None):
    allX,allY = createData()
    test_x = allX[160:]
    test_y = allY[160:]
#     test_x = allX[0:160]
#     test_y = allY[0:160]        
    out = evaluate(test_x,test_y)
    
    i=0
    count=0
    tempxx=0
    for i in range(test_y.shape[0]):
        if out[i][0] > out[i][1]:
            tempxx=0
        else:  
            tempxx=1   
        if test_y[i] == tempxx:
            count = count + 1
        i = i + 1
     
    print(count) 
    print(test_y.shape[0])    
if __name__ == '__main__':
    tf.app.run()    
                            
                          