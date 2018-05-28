import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import stockData

test_X,test_y = stockData.generate_data(240,250)

model = load_model('stockGRU5.h5')
predicted = model.predict(test_X)
predicted1 = predicted[220:250]
predicted1 = predicted1.flatten()
predicted1 = predicted1/10

test_y1 = test_y[220:250]
test_y1 = test_y1/10

test_X1 = test_X[220:250]
test_X1 = test_X1.flatten()
test_X1 = test_X1.reshape(int(test_X1.shape[0]/8), 8)
test_X1 = test_X1/10

rmse = np.sqrt(((predicted1 - test_y1)**2).mean(axis=0))
print('Mean SD Error is: %f' % rmse)

fig = plt.figure()
plot_predicted,  = plt.plot(predicted1, label='predicted')
plot_test,  = plt.plot(test_y1, label='real')
plt.legend([plot_predicted,plot_test],['predicted', 'real'])
plt.show()
