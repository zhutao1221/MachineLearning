import csv
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from keras.models import Sequential
from keras.layers import Dense,GRU
from keras.optimizers import Adam
import stockData

train_X,train_y = stockData.generate_data(0,240)
test_X,test_y = stockData.generate_data(0,245)

adam = Adam(lr=0.001)
model = Sequential()
model.add(GRU(100, activation='tanh',dropout= 0.01, return_sequences=True,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(GRU(100, activation='tanh',dropout= 0.01, return_sequences=True))
model.add(GRU(100, activation='tanh'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer=adam)
model.summary()

model.fit(train_X, train_y, epochs=30000, batch_size=100, verbose=1, shuffle=False)
model.save('stockGRU5.h5')

predicted = model.predict(test_X)
predicted = predicted.flatten()
predicted = predicted/10
test_y = test_y/10

rmse = np.sqrt(((predicted - test_y)**2).mean(axis=0))
print('Mean SD Error is: %f' % rmse)

fig = plt.figure()
plot_predicted,  = plt.plot(predicted, label='predicted')
plot_test,  = plt.plot(test_y, label='real')
plt.legend([plot_predicted,plot_test],['predicted', 'real'])
plt.show()

