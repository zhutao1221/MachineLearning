import os
import tensorflow as tf
import stockData
import stockInference
from tensorflow.contrib.nn.python.ops import cross_entropy
import numpy as np

BATCH_SIZE = 60
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 90
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = 'model'
MODEL_NAME = 'model.ckpt'

def train():
    sess = tf.Session()
    
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,
                           [BATCH_SIZE,
                            stockInference.Row_SIZE,
                            stockInference.Col_SIZE,
                            stockInference.NUM_CHANNELS], 
                            name='x-input')
    
        y_ = tf.placeholder(tf.float32, [None, stockInference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    #with tf.name_scope('output'):    
    y = stockInference.inference(x, 1, regularizer)
        
    global_step = tf.Variable(0, trainable=False)
    
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):    
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(y_, 1), logits = y)
    
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope('train_step'): 
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, 
            global_step, 
            5580 / BATCH_SIZE, 
            LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
        
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys  = stockData.allBatchData(BATCH_SIZE,6,i)
            #print(ys)           
            #xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          stockInference.Row_SIZE,
                                          stockInference.Col_SIZE,
                                          stockInference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict={x: reshaped_xs, y_:ys})
                       
            #print(sess.run(y,feed_dict={x: reshaped_xs, y_:ys}))
            print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
#     writer = tf.summary.FileWriter("/tmp/tflog", tf.get_default_graph())       
#     writer.close()   
        
def main(argv=None):
    train()

    
if __name__ == '__main__':
    tf.app.run()