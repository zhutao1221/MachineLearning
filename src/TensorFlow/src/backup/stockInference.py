import tensorflow as tf

INPUT_NODE = 1200
OUTPUT_NODE = 2
NUM_CHANNELS = 1

Row_SIZE = 6
Col_SIZE = 4
NUM_LABELS = 2

CONV1_DEEP = 32
CONV1_Row_SIZE = 3
CONV1_Col_SIZE = 1

CONV2_DEEP = 64
CONV2_Row_SIZE = 3
CONV2_Col_SIZE = 1

FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_Row_SIZE, CONV1_Col_SIZE, NUM_CHANNELS, CONV1_DEEP], 
                                        initializer=tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biases = tf.get_variable(
            'bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        #print('relu1:')
        #print(relu1)
        
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize = [1,3,1,1], strides=[1,3,1,1], padding='SAME')
        
        #print('pool1:')
        #print(pool1)        
        
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_Row_SIZE, CONV2_Col_SIZE, CONV1_DEEP, CONV2_DEEP], 
                                        initializer=tf.truncated_normal_initializer(stddev = 0.1))
        conv2_biases = tf.get_variable(
            'bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        #print('relu2:')
        #print(relu2)   
        
        
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize = [1,3,1,1], strides=[1,3,1,1], padding='SAME')
    
    pool_shape = pool2.get_shape().as_list()
    
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    #print('reshaped:')
    #print(reshaped)  
    
    
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable(
            'weight', [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
               
        #print('fc1:')
        #print(fc1)
          
    with tf.variable_scope('layer6_fc2'):
        fc2_weights = tf.get_variable(
            'weight', [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        #print('fc2_weights:')
        #print(fc2_weights)
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        
    return logit