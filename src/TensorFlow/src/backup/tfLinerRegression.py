import tensorflow as tf  
import numpy as np  
  
def createData():
    train_x = np.array([[1.0,1.0,1.0],[1.0,2.0,2.0],[2.0,2.0,3.0],[3.0,4.0,4.0]])
    train_y = np.array([[2.0],[4.0],[5.0],[9.0]])
    return train_x,train_y  
  
  
def linerRegression(train_x,train_y,epoch=900000,rate = 0.00001):
    
    with tf.variable_scope('input'): 
        n = train_x.shape[0]  
        x = tf.placeholder(tf.float32,name='x-input')  
        y = tf.placeholder(tf.float32,name='y-input')
    
    with tf.variable_scope('layer1_fc1'):      
        fc1_weights = tf.get_variable(
            'weight1', [3, 5],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        fc1_biases = tf.get_variable(
            'bias1', [5], initializer=tf.constant_initializer(0.1)) 
        fc1 = tf.nn.relu(tf.add(tf.matmul(x,fc1_weights),fc1_biases))
        
    with tf.variable_scope('layer2_fc2'):      
        fc2_weights = tf.get_variable(
            'weight2', [5, 3],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        fc2_biases = tf.get_variable(
            'bias2', [3], initializer=tf.constant_initializer(0.1)) 
        fc2 = tf.nn.relu(tf.add(tf.matmul(fc1,fc2_weights),fc2_biases))
        
    with tf.variable_scope('output'):      
        fc3_weights = tf.get_variable(
            'weight3', [3, 1],
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        fc3_biases = tf.get_variable(
            'bias3', [1], initializer=tf.constant_initializer(0.1)) 
        output = tf.add(tf.matmul(fc2,fc3_weights),fc3_biases)                  
        
    with tf.name_scope('loss_function'):          
        loss = tf.reduce_sum(tf.square(output-y))
    
    with tf.name_scope('train_step'):      
        optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)  
  
    init = tf.global_variables_initializer()  
  
    #sess = tf.Session()  
    #sess.run(init) 
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()     
        for index in range(epoch):
            print(index)     
            sess.run(optimizer,{x:train_x,y:train_y})
        
        
        print ('loss is ',sess.run(loss,{x:train_x,y:train_y}))
        print ('output is ',sess.run(output,{x:train_x,y:train_y}))    
        writer = tf.summary.FileWriter("/tmp/tflog", tf.get_default_graph())
        writer.close()    

        w1 =  sess.run(fc1_weights)  
        b1 = sess.run(fc1_biases)
        w2 =  sess.run(fc2_weights)  
        b2 = sess.run(fc2_biases)    
    return w1,b1,w2,b2 
  
# def predictionTest(test_x,test_y,w,b):  
#     W = tf.placeholder(tf.float32)  
#     B = tf.placeholder(tf.float32)  
#     X = tf.placeholder(tf.float32)  
#     Y = tf.placeholder(tf.float32)  
#     n = test_x.shape[0]  
#     pred = tf.add(tf.matmul(X,W),B)  
#     loss = tf.reduce_mean(tf.pow(pred-Y,2))  
#     sess = tf.Session()  
#     loss = sess.run(loss,{X:test_x,Y:test_y,W:w,B:b})  
#     return loss  
  
  
  
if __name__ == "__main__":  
    train_x,train_y = createData()  
    w1,b1,w2,b2 = linerRegression(train_x,train_y)  
    #loss = predictionTest(test_x,test_y,w,b)  
    #print (loss)
     