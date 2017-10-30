'''
@author: ShameQ
@github: https://github.com/ShameQ
'''

from math import log
import numpy as np

# 计算熵
def calcEnt(dataSet):
    m = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1]
        labelCounts[label] = labelCounts.get(label, 0) + 1

    Ent = 0
    for key in labelCounts:
        p = labelCounts[key] / m
        Ent -= p * log(p, 2)

    return Ent

# 根据特征和值分裂数据集
def splitDataSet(dataSet, axis, value):
    dataUse = dataSet[dataSet[:, axis]==value]
    res = np.hstack([dataUse[:, :axis], dataUse[: ,axis+1:]])
    return res

# 选择信息增益比最大的特征
def chooseBestFeatureToSplit(dataSet):
    n = dataSet.shape[1] - 1
    baseEnt = calcEnt(dataSet)
    bestGainRatio = 0
    bestFeature = -1
    for i in range(n):
        uniqueVals = set(dataSet[:, i])
        newEnt = 0
        splitInformation = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            p = len(subDataSet) / len(dataSet)
            newEnt += p * calcEnt(subDataSet)
            splitInformation -= p * np.log2(p)
        if splitInformation == 0:
            pass
        else:
            infoGain = baseEnt - newEnt
            gainRatio = infoGain / splitInformation
            if gainRatio > bestGainRatio:
                bestGainRatio = gainRatio
                bestFeature = i
    return bestFeature

# 取多数为叶节点的类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

# 递归建立树
def createTree(dataSet, labels):
    classList = dataSet[:, -1]
    if sum(classList==classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    uniqueVals = set(dataSet[:, bestFeat])
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 测试用数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return np.array(dataSet), labels

myData, labels = createDataSet()
t = createTree(myData, labels)
print(t)
