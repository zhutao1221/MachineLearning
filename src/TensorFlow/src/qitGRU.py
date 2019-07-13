'''
Created on 2019年7月13日

@author: zhutao
'''
import pickle
import numpy as np

with open('data/qitData/x/stock.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/stock.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('stock')
# print(y_np)
print(len(y_np))
####################################################
with open('data/qitData/x/白银.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/白银.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('白银')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/大豆.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/大豆.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('大豆')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/黄金.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/黄金.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('黄金')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/咖啡.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/咖啡.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('咖啡')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/人民币.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/人民币.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('人民币')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/日元.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/日元.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('日元')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/天然气.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/天然气.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('天然气')
# print(y_np)
print(len(y_np))

####################################################
with open('data/qitData/x/猪肉.txt', 'rb') as f:
    x = pickle.load(f)
    
with open('data/qitData/y/猪肉.txt', 'rb') as f:
    y = pickle.load(f)
    
x_np = np.array(x)
y_np = np.array(y)

print('猪肉')
# print(y_np)
print(len(y_np))