'''
@author: memojune
@github: https://github.com/ShameQ
'''

from numpy import *

# 使用单层决策树（树桩，stump）作为基本分类器
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArr = ones((dataMat.shap[0], 1))
    if threshIneq == 'lt':
        retArr[dataMat[:, dimen] <= threshVal] = -1
    else:
        retArr[dataMat[:, dimen] > threshVal] = -1
    return retArr

def buildStump(dataArr, classLabels, D):
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = dataMat.shape
    numSteps = 10; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf

    for i in range(n):
        rangeMin = dataMat[:, i].min(); rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, numSteps+1):

            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + stepSize * j
                predictedVal = stumpClassify(dataMat, i, threshVal, inequal)
                err = mat(ones(m, 1))
                err[predictedVal == labelMat] = 0
                weightedErr = D.T * err    # 加权后的Error
                if weightedErr < minError:
                    minError = weightedErr
                    bestClassEst = predictedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

# 基于单层决策树的AdaBoost训练
def adaBoostTrain(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m, n = dataArr.shape
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1-error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        aggErr = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errRate = aggErr / aggErr.sum()
        if errRate == 0:
            break

    return weakClassArr

# 分类函数
def adaClassify(dataArr, classifyArr):
    dataMat = mat(dataArr)
    m, n = dataMat.shape
    aggClassEst = mat(ones((m, 1)))
    for i in range(len(classifyArr)):
        classEst = stumpClassify(dataMat, classifyArr[i]['dim'],
                                 classifyArr[i]['thresh'],
                                 classifyArr[i]['ineq'])
        aggClassEst += classifyArr[i]['alpha'] * classEst
    return sign(aggClassEst)

