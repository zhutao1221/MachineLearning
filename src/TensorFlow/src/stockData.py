import csv,pandas,numpy

def createDate():
    print('createDate')
    csvTemp = open('data/000001.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)

    list = pandas.DataFrame()
    tempVarLast = 0.0
    tempVarNow = 0.0 
    
    i = 0
    for rowTemp in readTemp:
        if i == 0:
            tempVarLast = float(rowTemp['开盘'])
            tempVarNow = float(rowTemp['收盘'])
            changeRate = 100 * (tempVarNow - tempVarLast)/tempVarLast
                   
            if tempVarNow < tempVarLast:
                label = 0
            else:
                label = 1
                                 
        else: 
            tempVarNow = float(rowTemp['收盘'])    
            changeRate = 100 * (tempVarNow - tempVarLast)/tempVarLast
            
            if tempVarNow < tempVarLast:
                label = 0
            else:
                label = 1 
         
        if  changeRate <= -10.0:
            changeLabel = 0
        elif  changeRate > -10.0 and changeRate <= -5.0:
            changeLabel = 1
        elif  changeRate > -5.0 and changeRate <= -2.0:
            changeLabel = 2
        elif  changeRate > -2.0 and changeRate <= -1.0:
            changeLabel = 3
        elif  changeRate > -1.0 and changeRate <= -0.5:
            changeLabel = 4
        elif  changeRate > -0.5 and changeRate < 0.0:
            changeLabel = 5
        elif  changeRate >= 0.0 and changeRate <= 0.5:
            changeLabel = 6            
        elif  changeRate > 0.5 and changeRate <= 1:
            changeLabel = 7
        elif  changeRate > 1.0 and changeRate <= 2.0:
            changeLabel = 8
        elif  changeRate > 2.0 and changeRate <= 5.0:
            changeLabel = 9
        elif  changeRate > 5.0 and changeRate <= 10.0:
            changeLabel = 10
        elif  changeRate > 10.0:
            changeLabel = 11
        else:
            changeLabel = 12
            
        shakerRate = 100 * (float(rowTemp['最高']) - float(rowTemp['最低'])) / float(rowTemp['收盘'])                                                                   
        
        list = list.append(pandas.DataFrame({'时间':rowTemp['时间'],
                                             '开盘':rowTemp['开盘'],
                                             '最高':rowTemp['最高'],
                                             '最低':rowTemp['最低'],
                                             '收盘':rowTemp['收盘'],
                                             '成交量':rowTemp['成交量'],
                                             '涨跌幅':changeLabel,
                                             '震荡幅':shakerRate,
                                             '涨跌':label},index=[i]))
        tempVarLast =  float(rowTemp['收盘'])
        i = i + 1
    list.to_csv('data/stock.csv',encoding='utf-8')
    
def oneBatchData(days,index):
    csvTemp = open('data/stock.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    x = []
        
    i = 0
    for rowTemp in readTemp:
        if i >= index and i < index + days:
            x.append(float(rowTemp['涨跌']))            
            x.append(float(rowTemp['成交量']))
            x.append(float(rowTemp['涨跌幅']))             
            x.append(float(rowTemp['震荡幅']))
        if i == index + days:
            if int(rowTemp['涨跌']) == 1:
                y = [0,1]
            else:
                y = [1,0]      
        i = i + 1    
    return x,y

def allBatchData(batch,days,index):
    x = []
    y = []    
    for i in range(batch):       
        xTemp,yTemp = oneBatchData(days,index)
        x.append(xTemp)
        y.append(yTemp)
        i = i + 1
        index = index + i
    return x,y    
        
           

if __name__ == '__main__':
    #createDate() 
#     x_,y_ = oneBatchData(6,0)
#     reshaped_xs = numpy.reshape(x_, (6,4))
#     print(reshaped_xs)
    x_,y_ = allBatchData(2,12,0)
    reshaped_xs = numpy.reshape(x_, (2,12,4))
    print(y_)  
    