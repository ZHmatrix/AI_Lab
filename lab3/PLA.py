# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:36:07 2019

@author: 仲逊
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData(path):
    """
    从给定路径的csv数据集中加载数据
    Args:
        path：字符串 代表数据集绝对路径

    Returns:
        train,labels二元组
        train numpy数组，增广后的(补充x_0=1)训练数据
        labels numpy数组，处理后的训练集标签(转化为{+1,-1})
    """
    
    dataSet=pd.read_csv(path,header=None)
    #取出数据集标签
    labels=dataSet.pop(len(dataSet.columns)-1).values.tolist()
    #标签是0和1，转化成-1和1
    labels=np.array([-1 if label==0 else 1 for label in labels])
    #数据集第一列补x0=1
    dataSet.insert(0,'0',np.ones(len(dataSet)))
    train=dataSet.values
    
    return train,labels

def sign(x):
    return 1 if x>0 else -1

def PLAGradDescent(train,labels,alpha,cycle):
    """
    在给定训练集上执行感知机学习算法的梯度下降法
    Args:
        train numpy数组，增广后的(补充x0=1)训练数据
        labels numpy数组，处理后的训练集标签(只有{+1,-1})
        alpha 学习率
        cycle 感知机学习中遍历数据集的变数
    Returns:
        weights numpy矩阵类型 列向量
        例如：np.mat([[1],[2],[3],[4]])
        表示权重w为列向量(1,2,3,4)
    """
    
    #训练数据转化为矩阵形式
    trainMat = np.mat(train)
    #label转化成列向量             
    labelVec = np.mat(labels).transpose()
    sampleNum,featNum = np.shape(trainMat)
    
    #weights初始化为0，定义为列向量方便矩阵运算 
    weights = np.zeros((featNum,1))
    #遍历整个数据集cycle轮
    for i in range(cycle): 
        for j in range(sampleNum):
            #预测分类
            h=sign(trainMat[j]*weights)  
            #遇到误分类样本使用公式更新权重                  
            if h!=labelVec[j][0]:
                weights=weights+alpha*trainMat[j].transpose()*labelVec[j]
                break
    return weights

def KFold(dataSet,k):
    """
    进行K折交叉验证的数据集划分
    Args:
        dataSet numpy数组 含标签的数据集
        k 整型 将数据集划分为k个部分
    Returns：
        列表 每个元素代表数据集的一部分
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
########################全部训练全部测试#############################
#accuray=[]
#train,labels=loadData("D:/AI Lab/lab3/train.csv")
#for cycle in range(1,30):
#    weights=PLAGradDescent(train,labels,1,cycle)
#    predict=[sign(np.mat(sample)*weights) for sample in train]
#    correct=0
#    for i in range(len(predict)):
#        if(predict[i]==labels[i]):
#            correct+=1
#    accuray.append(correct/len(predict))
#    print("cycle",cycle,"accuracy:",correct/len(predict))
#plt.scatter(range(1,30), accuray)
#plt.title('PLA cycle-accuracy')
#plt.xlabel('cycle')
#plt.ylabel('accuracy')
#plt.grid(True)
#plt.show()
########################K折交叉验证#############################
#train,labels=loadData("D:/AI Lab/lab3/train.csv")
#k=5
#foldList=KFold(np.insert(np.array(train),len(train[0]),values=np.array(labels),axis=1),k)
#labelsList=[[labels[-1] for labels in fold] for fold in foldList]
#trainList=[np.delete(fold, -1, axis=1).tolist() for fold in foldList]
#
#avgAccurList=[]
#for cycle in range(1,30):    
#    avgAccur=0.0
#    for i in range(k):
#        trainListCopy=trainList.copy()
#        labelsListCopy=labelsList.copy()
#        
#        test=trainListCopy[i].copy()
#        del test[i]
#        test=np.array(test)
#        testLabels=labelsListCopy[i].copy()
#        del testLabels[i]
#        testLabels=np.array(testLabels)
#        
#        train=[]
#        labels=[]
#        for j in range(k):
#            if(j!=i):
#               train.extend(trainListCopy[j])
#               labels.extend(labelsListCopy[j])  
#        
#        train=np.array(train)
#        labels=np.array(labels)
#        weights=PLAGradDescent(train,labels,1,cycle)
#        predict=[sign(np.mat(sample)*weights) for sample in test]
#        
#        correct=0
#        for i in range(len(predict)):
#            if(predict[i]==testLabels[i]):
#                correct+=1
#        avgAccur+=correct/len(predict)
#    avgAccur/=k
#    avgAccurList.append(avgAccur)
#    print(k,"Fold Average accuracy：",avgAccur)
#    
#plt.scatter(range(1,30), avgAccurList)
#plt.title('PLA cycle 5-Fold Cross-validation accuracy')
#plt.xlabel('cycle')
#plt.ylabel('accuracy')
#plt.grid(True)
#plt.show() 
########################验收#############################
train,labels=loadData("D:/QQfile/check_train.csv")
cycle=100
weights=PLAGradDescent(train,labels,1,cycle)
predict=[sign(np.mat(sample)*weights) for sample in train]
correct=0
for i in range(len(predict)):
    if(predict[i]==labels[i]):
        correct+=1
print("train set accuracy:",correct/len(predict))

test,testLabels=loadData("D:/QQfile/check_test.csv")
predict=[sign(np.mat(sample)*weights) for sample in test]
print(predict)