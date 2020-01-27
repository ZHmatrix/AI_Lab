# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:25:13 2019

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:33:38 2019

@author: 仲逊
"""

import numpy as np
import pandas as pd


def loadData(path):
    train=pd.read_csv(path)
    labels=train.pop("Label").values.tolist()
    attributes=train.columns.values.tolist()
    train=train.values.tolist()
    return train,labels,attributes

def computeEnt(labels):
    """
    input：labels 数据集的标签
    return：数据集的信息熵
    """
    sampleNum = len(labels)
    #统计各个label出现次数
    labelCounts = {}
    for i in range(sampleNum): 
        labelCounts[labels[i]]=labelCounts.get(labels[i],0)+1
    #计算信息熵
    entropy = 0.0
    for label in labelCounts:
        prob = labelCounts[label]/sampleNum
        entropy -= prob * np.log2(prob) 
    return entropy

#def computeSplitInfo(attributeVec):
#    sampleNum = len(attributeVec)
#    #统计各个label出现次数
#    labelCounts = {}
#    for i in range(sampleNum): 
#        labelCounts[attributeVec[i]]=labelCounts.get(attributeVec[i],0)+1
#    #计算SplitInfo
#    splitInfo = 0.0
#    for key in labelCounts:
#        prob = labelCounts[key]/sampleNum
#        splitInfo -= prob * np.log2(prob) 
#    return splitInfo

def computeGini(attribute, labels):
    """
    input：attribute 数据集的一列
           label 数据集的标签
    return：选取该属性分裂时数据集的Gini指数
    """
    sampleNum = len(labels)
    #统计各个属性值的出现次数
    attriCounts = {}
    for i in range(sampleNum): 
        attriCounts[attribute[i]]=attriCounts.get(attribute[i],0)+1
    #计算信息熵
    gini = 0.0
    for key in attriCounts:
        labelCounts = {}
        #统计一个属性值对应的不同label数量
        for i in range(sampleNum): 
            if attribute[i] == key:
                labelCounts[labels[i]]=labelCounts.get(labels[i],0)+1
        #该属性值对应的权重       
        weight = attriCounts[key]/sampleNum
        #使用公式计算该属性的Gini指数
        attrGini = (1-sum([(labelCounts[label]/attriCounts[key])**2 for label in labelCounts]))
        gini +=  weight*attrGini
    return gini

def splitData(dataSet, labels, axis, value):
    retDataSet = []
    retLabels = []
    for i in range(len(dataSet)):
        featVec = dataSet[i]
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]    
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
            retLabels.append(labels[i])
    return retDataSet,retLabels

def ID3SplitFeature(train,labels):
    """
    input：train 训练数据
           labels 训练标签
    return：最佳分裂属性的下标(对应属性名列表中位置)
    """
    numFeatures = len(train[0])
    entropy = computeEnt(labels)
    bestmutualInfo = 0.0
    bestFeaIndex = -1
    for i in range(numFeatures):
        #取出一列(一个attribute)
        attribute = [sample[i] for sample in train]
        #获得特征的取值集合
        uniqueVals = set(attribute)  
        #计算在选取第i个特征下的条件熵
        conditionalEnt = 0.0      
        for value in uniqueVals:
            splitedData,splitedlabels = splitData(train, labels, i, value)
            prob = len(splitedData)/len(train)
            conditionalEnt += prob * computeEnt(splitedlabels)  
        #计算互信息（信息增益）
        mutualInfo = entropy - conditionalEnt 
        #选取最佳信息增益    
        if (mutualInfo > bestmutualInfo):       
            bestmutualInfo = mutualInfo         
            bestFeaIndex = i
    return bestFeaIndex                      

def C45SplitFeature(train,labels):
    """
    input：train 训练数据
           labels 训练标签
    return：最佳分裂属性的下标(对应属性名列表中位置)
    """
    numFeatures = len(train[0])
    entropy = computeEnt(labels)
    bestInfoGainRat = 0.0
    bestFeaIndex = -1
    for i in range(numFeatures):
        #取出一列(一个attribute)
        attribute = [sample[i] for sample in train]
        #获得特征的取值集合
        uniqueVals = set(attribute)  
        #计算在选取第i个特征下的条件熵
        conditionalEnt = 0.0      
        for value in uniqueVals:
            splitedData,splitedlabels = splitData(train, labels, i, value)
            prob = len(splitedData)/len(train)
            conditionalEnt += prob * computeEnt(splitedlabels)  
        #计算互信息和属性熵
        mutualInfo = entropy - conditionalEnt     
        splitInfo = computeEnt(attribute)
        #选取最佳信息增益率
        if bestInfoGainRat < mutualInfo/splitInfo:
            bestInfoGainRat = mutualInfo/splitInfo
            bestFeaIndex = i
    return bestFeaIndex 

def CARTSplitFeature(train,labels):
    """
    input：train 训练数据
           labels 训练标签
    return：最佳分裂属性的下标(对应属性名列表中位置)
    """
    numFeatures = len(train[0])
    bestGini = float('inf')
    bestFeaIndex = -1
    for i in range(numFeatures):
        #取出一列(一个attribute)
        attribute = [sample[i] for sample in train]
        gini = computeGini(attribute, labels)
        #选取最大信息增益
        if gini < bestGini:       
            bestGini = gini         
            bestFeaIndex = i

    return bestFeaIndex 


def buildTree(train,labels,attributes,attrValSet,chooseFeatFun):
    """
    input：train 训练数据
           labels 训练标签
           attributes 属性名列表 
           attrValSet 列表，每个元素代表每个属性的取值集合
           chooseFeatFun 选择最佳分裂属性的函数
           使用方法为chooseFeatFun(train,labels)
           得到的是最佳分裂属性的下标(对应属性名列表中位置)
    return：一棵用嵌套字典表示的决策树
            如{a:{'值1':{b:{'值1':1,'值2':0}},'值2':0} 表示的就是：
               |值2--分类0
            a--|      |值1--分类1
               |值1--b|
                      |值2--分类0
    """ 
    #如果数据集为空则表明该属性已经在之前被过滤掉，返回父节点标签的众数
    #此处返回空字典 用于接下来递归处理时的判断
    if len(train)==0:
        return {}
    #所有样本属于同一类别，将当前结点标记为该类别，返回该label
    if labels.count(labels[0]) == len(labels): 
        return labels[0]
    #用完了所有特征仍然不能将数据集划分成包含唯一类别的分组，此时返回label众数
    if len(train[0]) == 0: 
        return pd.Series(labels).mode()[0]

    #获取最佳分裂属性
    bestFeatIndex = chooseFeatFun(train,labels)
    bestFeatLabel = attributes[bestFeatIndex]
    #创建决策树，使用嵌套字典表示
    decisionTree = {bestFeatLabel:{}}
    del(attributes[bestFeatIndex])
    #获取最佳分裂属性中的属性值集合
    attrVals = attrValSet[bestFeatIndex]
    del attrValSet[bestFeatIndex]
    for value in attrVals:
        copyAttr = attributes.copy()
        copyattrValSet=attrValSet.copy()
        #以最佳分裂属性为根节点划分数据集
        splitedData,splitedlabels = splitData(train,labels,bestFeatIndex,value)
        #最佳分裂属性的一个值即为分支，分支再通过递归建树
        subTree = buildTree(splitedData,splitedlabels,copyAttr,copyattrValSet,chooseFeatFun)
        #如果子树为空字典，表明该值在此前已被过滤，返回父节点标签的众数
        if (not isinstance(subTree, dict)) or len(subTree)>0:
            decisionTree[bestFeatLabel][value] = subTree 
        else:
            decisionTree[bestFeatLabel][value] = pd.Series(labels).mode()[0]
    return decisionTree

def classify(decisionTree,attributes,testVec):
    """
    input：decisionTree 建成的决策树，由嵌套字典表示
           attributes 属性名列表
           test 一个测试数据
    return：预测结果
    """
    #字典的第一个键，对应一个属性
    firstAttr = list(decisionTree.keys())[0]
    #首键对应的值是个字典，字典对应的键就是属性的各个取值，即分支
    firstAttrBranch = decisionTree[firstAttr]
    featIndex = attributes.index(firstAttr)

    predictLabel=None
    #顺着分支递归查找，内部节点为字典，叶节点为预测标签
    for key in list(firstAttrBranch.keys()):
        if testVec[featIndex]==key:
            if isinstance(firstAttrBranch[key], dict):
                predictLabel = classify(firstAttrBranch[key], attributes, testVec)
            else:
                predictLabel = firstAttrBranch[key]
    return predictLabel
    
def KFold(dataSet,k):
    """
    input：dataSet 含标签的数据集，类型为np.array
           k 将数据集划分为k个部分
    return：列表 每个元素代表数据集的一部分
    """
    #获取打乱后的数据集下标
    rowIndex = np.arange(dataSet.shape[0])
    np.random.shuffle(rowIndex)
    foldSize = int(len(dataSet)/k)
    foldList = []
    counts=k
    #每次取int(len(dataSet)/k)个sample作为一个fold
    while(counts>1):
        foldList.append(dataSet[rowIndex[(k-counts)*foldSize:(k-counts+1)*foldSize]])
        counts-=1  
    foldList.append(dataSet[rowIndex[(k-1)*foldSize:]])

    return [fold.tolist() for fold in foldList]

#######################全部训练全部测试###########################
#path="D:/AI Lab/lab2/car_train.csv"
#train,labels,attributes=loadData(path)
#attrValSet=[set([sample[i] for sample in train]) for i in range(len(train[0]))]
#ans=buildTree(train.copy(),labels.copy(),attributes.copy(),attrValSet,C45SplitFeature)
#correct=0
#sampleNum=len(train)
#for i in range(sampleNum):
#    if(classify(ans,attributes,train[i])==labels[i]):
#        correct+=1
#print("accuracy：",correct/sampleNum)

#######################K折交叉验证###########################
#path="D:/AI Lab/lab2/car_train.csv"
#train,labels,attributes=loadData(path)
#k=5
#foldList=KFold(np.insert(np.array(train),len(train[0]),values=np.array(labels),axis=1),k)
#
#labelsList=[[labels[-1] for labels in fold] for fold in foldList]
#trainList=[np.delete(fold, -1, axis=1).tolist() for fold in foldList]
#
#avgAccur=0.0
#for i in range(k):
#    trainListCopy=trainList.copy()
#    labelsListCopy=labelsList.copy()
#    test=trainListCopy[i].copy()
#    del test[i]
#    testLabels=labelsListCopy[i].copy()
#    del testLabels[i]
#    train=[]
#    labels=[]
#    for j in range(k):
#        if(j!=i):
#           train.extend(trainListCopy[j])
#           labels.extend(labelsListCopy[j])  
#           
#    attrValSet=[set([sample[i] for sample in train]) for i in range(len(train[0]))]
#    ans=buildTree(train.copy(),labels.copy(),attributes.copy(),attrValSet,ID3SplitFeature)
#    correct=0
#    sampleNum=len(test)
#    for i in range(sampleNum):
#        if(classify(ans,attributes,test[i])==testLabels[i]):
#            correct+=1
#    print(correct/sampleNum)
#    avgAccur+=correct/sampleNum
#    
#avgAccur/=k
#print(k,"Fold Average accuracy：",avgAccur)
    
#######################验收###########################
path="D:/QQfile/train.csv"
train,labels,attributes=loadData(path)
attrValSet=[set([sample[i] for sample in train]) for i in range(len(train[0]))]
ans=buildTree(train.copy(),labels.copy(),attributes.copy(),attrValSet,ID3SplitFeature)
correct=0
sampleNum=len(train)
for i in range(sampleNum):
    if(classify(ans,attributes,train[i])==labels[i]):
        correct+=1
#        print(correct)
print(correct/sampleNum)

path="D:/QQfile/test.csv"
test,labels,attributes=loadData(path)
for i in range(len(test)):
    print(classify(ans,attributes,test[i]))