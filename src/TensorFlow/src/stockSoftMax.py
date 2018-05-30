from keras.models import Sequential
from keras.layers import Dense,GRU
from keras.optimizers import Adam
import stockData

train_X,train_y = stockData.softmax_data(0,269)
test_X,test_y = stockData.softmax_data(0,269)

adam = Adam(lr=0.001)
model = Sequential()
model.add(GRU(100, activation='tanh',dropout= 0.01, return_sequences=True,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(GRU(100, activation='tanh',dropout= 0.01, return_sequences=True))
model.add(GRU(100, activation='tanh'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.summary()
print(train_X.shape)
print(train_y.shape)
model.fit(train_X, train_y, epochs=10000, batch_size=100, verbose=1, shuffle=False)
model.save('softmax.h5')

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
       