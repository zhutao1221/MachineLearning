import os
import tensorflow as tf
import sonar_inference
from tensorflow.contrib.nn.python.ops import cross_entropy
import numpy as np
import csv

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

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

def train(xTrain,yTrain):
    
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,name='x-input')
        tf.add_to_collection('predX', x)       
        y_ = tf.placeholder(tf.int32,name='y-input')  

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    #with tf.name_scope('output'):    
    y = sonar_inference.inference(x, 0, regularizer)
    tf.add_to_collection('predY', y)
        
    global_step = tf.Variable(0, trainable=False)
    
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):
        #print(y)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = y)
    
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope('train_step'): 
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, 
            global_step, 
            20, 
            LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
        
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):  
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict={x: xTrain, y_:yTrain})
            print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
        saver.save(sess, 'model/model.ckpt')
        #print(sess.run(y_,feed_dict={x: xTrain, y_:yTrain})) 
    writer = tf.summary.FileWriter("/tmp/tflog", tf.get_default_graph())        
    writer.close()   
        
def main(argv=None):
    allX,allY = createData()
    train_x = allX[0:160]
    train_y = allY[0:160]    
    out = train(train_x,train_y)        
    
 
if __name__ == '__main__':
    tf.app.run()