#import tensorflow as tf
import numpy as np

# xs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# 
# reshaped_xs = np.reshape(xs, (2,5,2))
# 
# print(reshaped_xs)
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print(np.argmax(test, 0)) #输出：array([3, 3, 1]
print(np.argmax(test, 1)) #输出：array([2, 2, 0, 0]