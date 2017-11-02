'''
@author: memojune
@github: https://github.com/ShameQ
'''

import numpy as np
from math import log

# 计算先验概率和条件概率
def trainNB0(trainMatrix, trainCategory):
    m = len(trainMatrix)
    n = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / m
    p0Num = np.zeros(n); p1Num = np.zeros(n)
    p0Denom = 0; p1Denom = 0

    for i in range(m):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    return p0Vec, p1Vec, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0






















