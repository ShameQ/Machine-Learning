'''
@author: memojune
@github: https://github.com/ShameQ
'''

import numpy as np
from math import exp
import random

def sigmoid(x):
    return 1 / (1 + exp(-x))

# 标准梯度下降
def gradAscent(dataSet, classLabels):
    dataMat = np.mat(dataSet)
    labelMat = np.mat(classLabels).transpose()
    m, n = dataMat.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for i in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error
    return weights

# 随机梯度下降
def stocGradAscent(dataMat, classLabels, numIter=150):
    m, n = dataMat.shape
    weights = np.ones((n, 1))
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1+i+j) + 0.01
            randIndex = dataIndex[random.randint(0, len(dataIndex)-1)]
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights
























