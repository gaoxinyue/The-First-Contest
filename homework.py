import pandas as pd
import numpy as np
import operator

# 读取两个文件中的数据并返回经处理后的数据
def initData():
    # 首先调用processData函数对训练文件HTRU_2_train.csv中的数据进行处理
    data1 = processData()
    # 读取测试文件HTRU_2_test.csv中的数据
    data2 = np.loadtxt('HTRU_2_test.csv', delimiter=',')
    trainSet = data1
    testSet = data2
    # trainData存放训练集前两列（即不包括类别）
    trainData = trainSet[:, 0:-1]
    # testData存放测试集所有数据（即两列）
    testData= testSet[:, :]
    # trainClass存放训练集最后一列的类别信息（即0或1）
    trainClass = trainSet[:, -1]
    return trainData, testData, trainClass

# 对HTRU_2_train.csv文件中的数据进行处理
def processData():
    data1 = np.loadtxt('HTRU_2_train.csv', delimiter=',')
    # 第一列属性
    a = data1[:, 0]
    b = list(a)
    # 调用findDeleteIndex函数找到data1数据中符合要求可以进行删除的索引
    index = findDeleteIndex(b)
    # 删除index中的索引返回给trainSet
    data2 = np.delete(data1, index, axis=0)
    
    return data2

# 返回列表中符合要求的元素对应索引所组成的列表
def findDeleteIndex(b):
    num = 0
    arr = []
    # Q1表示在b列表数据中从小到大占25%位置的元素
    Q1 = np.percentile(b, 25)
    # Q3表示在b列表数据中从小到大占75%位置的元素
    Q3 = np.percentile(b, 75)
    # 第75%和第25%的元素中间相差的数字
    IQR = Q3 - Q1
    step = 0.3 * IQR
    # 遍历b列表中的所有元素
    for nu in b:
        # 若b列表中的元素比Q1-step小或比Q3+step大，arr就记录该元素的索引号
        if (nu < Q1 - step) | (nu > Q3 + step):
            arr.append(num)
        num = num + 1
    
    return arr

# 使用k近邻算法，返回测试集的类别
def knn(inX, dataSet,labels, k):
    # 获取标签向量的元素数目（即dataSet的行数）
    dataSetSize = dataSet.shape[0]
    # 使用欧式距离公式计算inX到各dataSet的距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances .argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对数据按照从小到大的次排序，确定前k个最小元素所在的主要分类
    # 将classCount字典分解为元组列表，然后使用itemgetter(1)按照第二个元素的从大到小次序
    # 对元组进行排序，最后返回发生频率最高的元素标签
    sortedClassCount = sorted(classCount.items(), 
    key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 分别为训练集前两列，测试集所有列，训练集最后一列的类别
    trainData, testData, trainClass = initData()
    # 接受测试集的预测类别
    testPredList = []
    for i in testData:
        testSet = [i[0], i[1]]
        # 采用k近邻算法，其中的参数k为16，表示选择最近邻的数目为16
        ret = knn(testSet, trainData, trainClass, 16)
        testPredList.append(ret)
    
    testPredList = np.array(testPredList)
    # 将类别转化为整数0或1
    testPredList = testPredList.astype(np.int)
    data = pd.DataFrame(testPredList, index=range(1, 701))
    data.to_csv('testPredict.csv')
    








