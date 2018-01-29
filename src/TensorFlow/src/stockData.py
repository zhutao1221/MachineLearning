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
            changeRate = 0.0      
        else: 
            tempVarNow = float(rowTemp['收盘'])    
            changeRate = (tempVarNow - tempVarLast)/tempVarLast
            changeRate = changeRate * 100 
        
        if  changeRate <= -10.0:
            label = 1
        elif  changeRate > -10.0 and changeRate <= -5.0:
            label = 2
        elif  changeRate > -5.0 and changeRate <= -2.0:
            label = 3
        elif  changeRate > -2.0 and changeRate <= -1.0:
            label = 4
        elif  changeRate > -1.0 and changeRate <= -0.5:
            label = 5
        elif  changeRate > -0.5 and changeRate <= 0.5:
            label = 6
        elif  changeRate > 0.5 and changeRate <= 1:
            label = 7
        elif  changeRate > 1.0 and changeRate <= 2.0:
            label = 8
        elif  changeRate > 2.0 and changeRate <= 5.0:
            label = 9
        elif  changeRate > 5.0 and changeRate <= 10.0:
            label = 10
        elif  changeRate > 10.0:
            label = 11
        else:
            label = 0                                                                
        
        list = list.append(pandas.DataFrame({'时间':rowTemp['时间'],
                                             '开盘':rowTemp['开盘'],
                                             '最高':rowTemp['最高'],
                                             '最低':rowTemp['最低'],
                                             '收盘':rowTemp['收盘'],
                                             '成交量':rowTemp['成交量'],
                                             '涨跌幅':changeRate,
                                             '标签':label},index=[i]))
        tempVarLast =  float(rowTemp['收盘'])    
        i = i + 1
    list.to_csv('data/stock.csv',encoding='utf-8')
    
def oneBatchData(days,index):
    csvTemp = open('data/stock.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    x = []
    y = []
    
    i = 0
    for rowTemp in readTemp:
        if i >= index and i < index + days:
            x.append(float(rowTemp['开盘']))
            x.append(float(rowTemp['收盘']))
            x.append(float(rowTemp['最高']))
            x.append(float(rowTemp['最低']))
            x.append(float(rowTemp['成交量']))
            y.append(int(rowTemp['标签']))
        i = i + 1    
    return x,y

def allBatchData(batch,days,index):
    for i in range(batch):
        xTemp,yTemp_ = oneBatchData(days,index)
        i = i + 1
        index = index + i
           

if __name__ == '__main__':
    #createDate() 
#     x_,y_ = nextBatch(6,0,0)
#     reshaped_xs = numpy.reshape(x_, (6,5))
#     print(reshaped_xs) 
    