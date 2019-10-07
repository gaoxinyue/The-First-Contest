import pandas as pd
import numpy as np
import operator

def initData():
    data1 = np.loadtxt('HTRU_2_train.csv', delimiter=',')
    data2 = np.loadtxt('HTRU_2_test.csv', delimiter=',')
    trainSet = data1
    testSet = data2
    trainData = trainSet[:, 0:-1]
    testData= testSet[:, :]
    trainClass = trainSet[:, -1]
    return trainData, testData, trainClass

def knn(inX, dataSet,labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances .argsort()
    
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), 
    key = operator.itemgetter(1), reverse = True)
#    print(sortedClassCount[0][0])
    return sortedClassCount[0][0]


if __name__ == '__main__':
    trainData, testData, trainClass = initData()
#    trainClass = trainClass.reshape(2652, 1)
    testPredList = []
    for i in testData:
        testSet=[i[0],i[1]]
        ret=knn(testSet,trainData,trainClass, 60)
        testPredList.append(ret)
    
    testPredList = np.array(testPredList)
    testPredList = testPredList.astype(np.int)
    data = pd.DataFrame(testPredList, index=range(1, 701))
    data.to_csv('data6.csv')
    








