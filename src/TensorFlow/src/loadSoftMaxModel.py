import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import stockData

test_X,test_y = stockData.softmax_data(260,261)

model = load_model('softmax.h5')
predicted = model.predict(test_X)

temp1 = []
for index in predicted:
    if index[0] > index[1]:
        temp1.append([0])
    else:
        temp1.append([1])    

temp2 = []
for index in test_y:
    if index[0] > index[1]:
        temp2.append([0])
    else:
        temp2.append([1])
        
countFit = 0
countAll = len(temp1)

i = 0
for i in range(countAll):
    if temp1[i] == temp2[i]:
        countFit = countFit + 1
        
print('countFit:' + str(countFit))        
print('countAll:' + str(countAll)) 