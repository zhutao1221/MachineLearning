import numpy as np
import csv
from math import exp
import stockIndex

def generate_data(startPoint,endPoint):
    allData = []
    csvTemp = open('data/stock.csv','r',encoding='utf-8')
    readTemp = csv.DictReader(csvTemp)
    
    endTemp5 = []
    endTemp10 = []
    endTemp20 = []
    endTemp30 = []
    
    doneTemp = []