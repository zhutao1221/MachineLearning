import tensorflow as tf

INPUT_NODE = 11

NUM_LABELS = 2

FC_SIZE1 = 100
FC_SIZE2 = 50
FC_SIZE3 = 10

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1_fc1'):
        fc1_weights = tf.get_variable(
            'weight', [INPUT_NODE, FC_SIZE1],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE1], initializer=tf.constant_initializer(0.1))
        
        fc1 = tf.nn.relu(tf.matmul(input_tensor, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
            
    with tf.variable_scope('layer2_fc2'):
        fc2_weights = tf.get_variable(
            'weight', [FC_SIZE1, FC_SIZE2],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
            
        fc2_biases = tf.get_variable(
            'bias', [FC_SIZE2], initializer=tf.constant_initializer(0.1))
        
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)
         
    with tf.variable_scope('layer3_fc3'):
        fc3_weights = tf.get_variable(
            'weight', [FC_SIZE2, FC_SIZE3],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
            
        fc3_biases = tf.get_variable(
            'bias', [FC_SIZE3], initializer=tf.constant_initializer(0.1))
        
        fc3 = tf.nn.relu(tf.matmul(fc2, fc3_weights) + fc3_biases)
        if train:
            fc3 = tf.nn.dropout(fc3, 0.5)                        

          
    with tf.variable_scope('layer4_fc4'):
        fc4_weights = tf.get_variable(
            'weight', [FC_SIZE3, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc4_weights))
        fc4_biases = tf.get_variable(
            'bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc3, fc4_weights) + fc4_biases
        
    return logit
