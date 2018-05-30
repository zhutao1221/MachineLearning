import numpy as np
import csv
from math import exp
import stockIndex

def generate_data(startPoint,endPoint):
    allData = []
    csvTemp = open('data/stockhnxj.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    endTemp5 = []
    endTemp10 = []
    endTemp20 = []
    endTemp30 = []    
    doneTemp = []
    
    kYesterday = 0.0
    DYesterday = 0.0
    
    flag = 0
    for rowTemp in readTemp:
        if len(endTemp5) == 5:
            endTemp5.pop(0)
            doneTemp.pop(0)
        if len(endTemp10) == 10:
            endTemp10.pop(0)
        if len(endTemp30) == 30:
            endTemp30.pop(0)
            
        price = 10*float(rowTemp['收盘'])
#         price = float(rowTemp['收盘'])
        endTemp5.append(price)
        endTemp10.append(price)
        endTemp30.append(price)
        doneTemp.append(float(rowTemp['成交量'])/100000000)
    
        endPrice = float(rowTemp['收盘'])
        lowPrice = float(rowTemp['最低'])
        highPrice = float(rowTemp['最高'])
    
        if flag == 0:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,'Null','Null')
            kYesterday = kt
            DYesterday = dt
            flag = 1
        else:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,kYesterday,DYesterday)
            kYesterday = kt
            DYesterday = dt
            
        if len(endTemp30) == 30:
            endlistTemp5 = np.array(endTemp5, dtype=np.float32)    
            endlistTemp10 = np.array(endTemp10, dtype=np.float32)                
            endlistTemp30 = np.array(endTemp30, dtype=np.float32)
            donelistTemp = np.array(doneTemp, dtype=np.float32)
            allData.append([endlistTemp5.mean(),price,endlistTemp10.mean(),endlistTemp30.mean(),kt,dt,jt,1.0/(1.0 + exp(-donelistTemp.mean()))])
            #allData.append([endlistTemp5.mean(),price,endlistTemp10.mean(),endlistTemp30.mean(),kt,dt,jt])
    XTemp = []
    yTemp = []
   
    XTemp = allData[startPoint:endPoint]
    yTemp = allData[startPoint+1:endPoint+1]
    
    X = np.array(XTemp, dtype=np.float32)
    y = np.array(yTemp, dtype=np.float32)
    y = y[:,0]
    
    X = X.flatten()
    X = X.reshape(int(X.shape[0]/8),1,8)
    #X = X.reshape(int(X.shape[0]/7),1,7)    
    y = y.flatten()
    return X,y 
    
def generate_data1(startPoint,endPoint):
    allData = []
    csvTemp = open('data/stockhnxj.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    endTemp3 = []
    endTemp5 = []
    endTemp10 = []
    
    kYesterday = 0.0
    DYesterday = 0.0
    
    flag = 0
    for rowTemp in readTemp:
        if len(endTemp3) == 3:
            endTemp3.pop(0)
        if len(endTemp5) == 5:
            endTemp5.pop(0)
        if len(endTemp10) == 10:
            endTemp10.pop(0)
            
        price = 10*float(rowTemp['收盘'])
#         price = float(rowTemp['收盘'])
        endTemp3.append(price)
        endTemp5.append(price)
        endTemp10.append(price)
        doneTemp = float(rowTemp['成交量'])/100000000
    
        endPrice = float(rowTemp['收盘'])
        lowPrice = float(rowTemp['最低'])
        highPrice = float(rowTemp['最高'])
    
        if flag == 0:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,'Null','Null')
            kYesterday = kt
            DYesterday = dt
            flag = 1
        else:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,kYesterday,DYesterday)
            kYesterday = kt
            DYesterday = dt
            
        if len(endTemp10) == 10:
            endlistTemp3 = np.array(endTemp3, dtype=np.float32)    
            endlistTemp5 = np.array(endTemp5, dtype=np.float32)                
            endlistTemp10 = np.array(endTemp10, dtype=np.float32)
            allData.append([price,endlistTemp3.mean(),endlistTemp5.mean(),endlistTemp10.mean(),kt,dt,jt,1.0/(1.0 + exp(-doneTemp))])
            #allData.append([endlistTemp3.mean(),price,endlistTemp5.mean(),endlistTemp10.mean(),kt,dt,jt])
    XTemp = []
    yTemp = []
   
    XTemp = allData[startPoint:endPoint]
    yTemp = allData[startPoint+1:endPoint+1]
    
    X = np.array(XTemp, dtype=np.float32)
    y = np.array(yTemp, dtype=np.float32)
    y = y[:,0]
    
    X = X.flatten()
    X = X.reshape(int(X.shape[0]/8),1,8)
    #X = X.reshape(int(X.shape[0]/7),1,7)    
    y = y.flatten()
    return X,y

def generate_data5(startPoint,endPoint):
    allData = []
    csvTemp = open('data/600160.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    endTemp3 = []
    endTemp5 = []
    endTemp10 = []    
    doneTemp = []
    
    kYesterday = 0.0
    DYesterday = 0.0
    
    flag = 0
    for rowTemp in readTemp:
        if len(endTemp3) == 3:
            endTemp3.pop(0)
            doneTemp.pop(0)
        if len(endTemp5) == 5:
            endTemp5.pop(0)
        if len(endTemp10) == 10:
            endTemp10.pop(0)
            
        price = 10*float(rowTemp['收盘'])
#         price = float(rowTemp['收盘'])
        endTemp3.append(price)
        endTemp5.append(price)
        endTemp10.append(price)
        doneTemp.append(float(rowTemp['成交量'])/100000000)
    
        endPrice = float(rowTemp['收盘'])
        lowPrice = float(rowTemp['最低'])
        highPrice = float(rowTemp['最高'])
    
        if flag == 0:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,'Null','Null')
            kYesterday = kt
            DYesterday = dt
            flag = 1
        else:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,kYesterday,DYesterday)
            kYesterday = kt
            DYesterday = dt
            
        if len(endTemp10) == 10:
            endlistTemp3 = np.array(endTemp3, dtype=np.float32)    
            endlistTemp5 = np.array(endTemp5, dtype=np.float32)                
            endlistTemp10 = np.array(endTemp10, dtype=np.float32)
            donelistTemp = np.array(doneTemp, dtype=np.float32)
            allData.append([endlistTemp5.mean(),price,endlistTemp3.mean(),endlistTemp10.mean(),kt,dt,jt,1.0/(1.0 + exp(-donelistTemp.mean()))])
            #allData.append([endlistTemp5.mean(),price,endlistTemp3.mean(),endlistTemp10.mean(),kt,dt,jt])
    XTemp = []
    yTemp = []
   
    XTemp = allData[startPoint:endPoint]
    yTemp = allData[startPoint+1:endPoint+1]
    
    X = np.array(XTemp, dtype=np.float32)
    y = np.array(yTemp, dtype=np.float32)
    y = y[:,0]
    
    X = X.flatten()
    X = X.reshape(int(X.shape[0]/8),1,8)
    #X = X.reshape(int(X.shape[0]/7),1,7)    
    y = y.flatten()
    return X,y

def generate_predict_data5():
    allData = []
    csvTemp = open('data/600160.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    endTemp3 = []
    endTemp5 = []
    endTemp10 = []    
    doneTemp = []
    
    kYesterday = 0.0
    DYesterday = 0.0
    
    flag = 0
    for rowTemp in readTemp:
        if len(endTemp3) == 3:
            endTemp3.pop(0)
            doneTemp.pop(0)
        if len(endTemp5) == 5:
            endTemp5.pop(0)
        if len(endTemp10) == 10:
            endTemp10.pop(0)
            
        price = 10*float(rowTemp['收盘'])
#         price = float(rowTemp['收盘'])
        endTemp3.append(price)
        endTemp5.append(price)
        endTemp10.append(price)
        doneTemp.append(float(rowTemp['成交量'])/100000000)
    
        endPrice = float(rowTemp['收盘'])
        lowPrice = float(rowTemp['最低'])
        highPrice = float(rowTemp['最高'])
    
        if flag == 0:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,'Null','Null')
            kYesterday = kt
            DYesterday = dt
            flag = 1
        else:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,kYesterday,DYesterday)
            kYesterday = kt
            DYesterday = dt
            
        if len(endTemp10) == 10:
            endlistTemp3 = np.array(endTemp3, dtype=np.float32)    
            endlistTemp5 = np.array(endTemp5, dtype=np.float32)                
            endlistTemp10 = np.array(endTemp10, dtype=np.float32)
            donelistTemp = np.array(doneTemp, dtype=np.float32)
            allData.append([endlistTemp5.mean(),price,endlistTemp3.mean(),endlistTemp10.mean(),kt,dt,jt,1.0/(1.0 + exp(-donelistTemp.mean()))])
            #allData.append([endlistTemp5.mean(),price,endlistTemp3.mean(),endlistTemp10.mean(),kt,dt,jt])
    XTemp = []   
    XTemp = allData[-1]    
    X = np.array(XTemp, dtype=np.float32)    
    X = X.flatten()
    X = X.reshape(int(X.shape[0]/8),1,8)
    return X

def softmax_data(startPoint,endPoint):
    allDataX = []
    allDateY = [] 
    csvTemp = open('data/600160.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    endTemp3 = []
    endTemp5 = []
    endTemp10 = []    
    doneTemp = []
    
    kYesterday = 0.0
    DYesterday = 0.0
    priceYest = 0.0
    
    flag = 0
    for rowTemp in readTemp:
        if len(endTemp3) == 3:
            endTemp3.pop(0)
            doneTemp.pop(0)
        if len(endTemp5) == 5:
            endTemp5.pop(0)
        if len(endTemp10) == 10:
            endTemp10.pop(0)
            
        priceToday = 10*float(rowTemp['收盘'])
#         price = float(rowTemp['收盘'])
        endTemp3.append(priceToday)
        endTemp5.append(priceToday)
        endTemp10.append(priceToday)
        doneTemp.append(float(rowTemp['成交量'])/100000000)
    
        endPrice = float(rowTemp['收盘'])
        lowPrice = float(rowTemp['最低'])
        highPrice = float(rowTemp['最高'])
    
        if flag == 0:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,'Null','Null')
            kYesterday = kt
            DYesterday = dt
            flag = 1
        else:
            kt,dt,jt = stockIndex.nKDJ(endPrice,lowPrice,highPrice,kYesterday,DYesterday)
            kYesterday = kt
            DYesterday = dt
            
        if len(endTemp10) == 10:
            endlistTemp3 = np.array(endTemp3, dtype=np.float32)    
            endlistTemp5 = np.array(endTemp5, dtype=np.float32)                
            endlistTemp10 = np.array(endTemp10, dtype=np.float32)
            donelistTemp = np.array(doneTemp, dtype=np.float32)
            
            if priceYest < priceToday:
                allDateY.append([1,0])
            else:
                allDateY.append([0,1])
#             allDataX.append([priceToday,endlistTemp3.mean(),endlistTemp5.mean(),endlistTemp10.mean(),kt,dt,jt,\
#                              1.0/(1.0 + exp(-donelistTemp.mean()))])
            allDataX.append([priceToday,endlistTemp3.mean(),endlistTemp5.mean(),endlistTemp10.mean(),kt,dt,jt])
        priceYest = priceToday   
    XTemp = []
    yTemp = []
   
    XTemp = allDataX[startPoint:endPoint]
    yTemp = allDateY[startPoint + 1:endPoint + 1]
    
    X = np.array(XTemp, dtype=np.float32)
    y = np.array(yTemp, dtype=np.float32)
    
    X = X.flatten()
    X = X.reshape(int(X.shape[0]/7),1,7)
    return X,y   