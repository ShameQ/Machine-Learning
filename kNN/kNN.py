'''
@author: memojune
@github: https://github.com/ShameQ
'''

import numpy as np

def classify(inX, dataSet, labels, k):
    m = dataSet.shape[0]
    diffMat = dataSet - inX
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)

    return sortedClassCount[0][0]

# 对数据做归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    res = (dataSet - minVals) / (maxVals - minVals)
    return res






