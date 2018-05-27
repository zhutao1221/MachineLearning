import numpy as np
import csv

def nKDJ(endPrice,lowPrice,highPrice,kYes,dYes):
    if highPrice - lowPrice == 0.0:
        RSV = 1.0
    else:
        RSV = (endPrice - lowPrice)/(highPrice - lowPrice)
    if kYes == 'Null' or dYes == 'Null':
        K = (2/3) * 50 + (1/3) * RSV
        D = (2/3) * 50 + (1/3) * K
    else:
        K = (2/3) * kYes + (1/3) * RSV
        D = (2/3) * dYes + (1/3) * K
    J = 3 * K - 2 * D
    return K,D,J           