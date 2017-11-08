'''
@author: memojune
@github: https://github.com/ShameQ
'''

from numpy import *

# 距离度量函数，可选择其他
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

# 随机生成簇中心
def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = zeros((k, n))
    for i in range(n):
        minVal = dataSet[:, i].min()
        rangeVal = float(dataSet[:, i].max() - minVal)
        centroids[:, i] = (minVal + rangeVal * random.rand(k, 1)).flatten()
    return centroids

# kMeans，可能陷入局部最优
def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = zeros((m, 2))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k): # 寻找最近的质心
                dist = distMeans(centroids[j, :], dataSet[i, :])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = False
            clusterAssment[i, ] = minIndex, minDist
        for i in range(k): # 更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0]==i)[0]]
            centroids[i, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分K-均值聚类算法
def biKmeans(dataSet, k, distMeans=distEclud):
    m = dataSet.shape[0]
    clusterAssment = zeros((m, 2))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    while len(centList) < k:
        minSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0]==i)[0]]
            centroidMat, splitClusterAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClusterAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0]!=i)[0], 1])
            if (sseSplit + sseSplit) < minSSE:
                minSSE = sseSplit + sseNotSplit
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClusterAss.copy()
        bestClustAss[nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0]
        centList.append(bestNewCents[1])
        clusterAssment[nonzero(clusterAssment[:, 0]==bestCentToSplit)] = bestClustAss
    return centList, clusterAssment
























