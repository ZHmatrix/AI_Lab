# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:24:00 2019

@author: acer
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
        train numpy数组，增广后的(补充x0=1)训练数据
        labels numpy数组，训练集标签
    """
    dataSet=pd.read_csv(path,header=None)
    #取出数据集标签
    labels=dataSet.pop(len(dataSet.columns)-1).values
    #数据集第一列补x0=1
    dataSet.insert(0,'0',np.ones(len(dataSet)))
    train=dataSet.values
    return train,labels

def sigmoid(x):
    return 1/(1+np.exp(-x))

def LRGradDescent(train,labels,alpha=0.001,cycle=500):
    """
    在给定训练集上执行logistics回归交叉熵损失函数的梯度下降法
    Args:
        train numpy数组，增广后的(补充x_0=1)训练数据
        labels numpy数组，训练集标签
        alpha 学习率
        cycle 梯度下降法的迭代次数
    Returns:
        weights numpy矩阵类型 列向量
        例如：np.mat([[1],[2],[3],[4]])
        表示权重w为列向量(1,2,3,4)
    """
    #训练数据转化为矩阵形式
    trainMat = np.mat(train)
    #labels转化成列向量，方便矩阵运算              
    labelVec = np.mat(labels).transpose()
    sampleNum,featNum = np.shape(trainMat)
    
    #weights初始化为0，定义为列向量方便矩阵运算  
    weights = np.zeros((featNum,1))
    for i in range(cycle):   
        #预测概率值               
        h=sigmoid(trainMat*weights) 
        #和实际的误差
        err=labelVec-h  
        #梯度下降到指定阈值可以认为收敛，退出循环  
        gradL2Norm=np.linalg.norm(trainMat.transpose()*err)/sampleNum        
        if  gradL2Norm< 1e-4:
            break
#        if i<=4 or i>=(cycle-5):
#            print("Gradient L-2 Norm:",gradL2Norm)
        #logistics回归 交叉熵损失函数 批梯度下降法公式
        weights=weights+alpha/sampleNum*trainMat.transpose()*err
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

#######################全部训练全部测试###########################
#train,labels=loadData("D:/AI Lab/lab3/train.csv")
#accuray=[]
#for cycle in range(100,2000,100):
#    weights=LRGradDescent(train,labels,0.1,cycle)
#    predict=np.array(sigmoid(train*weights).transpose())[0]
#    predict=[0 if x<0.5 else 1 for x in predict]
#    correct=0
#    for i in range(len(predict)):
#        if(predict[i]==labels[i]):
#            correct+=1
#    accuray.append(correct/len(predict))
#    print(correct/len(predict))
#plt.scatter(range(100,2000,100), accuray)
#plt.title('LR cycle-accuracy')
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
#for cycle in range(200,4000,200):    
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
#        weights=LRGradDescent(train,labels,0.1,cycle)
#        predict=np.array(sigmoid(test*weights).transpose())[0]
#        predict=[0 if x<0.5 else 1 for x in predict]
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
#plt.scatter(range(200,4000,200), avgAccurList)
#plt.title('LR cycle 5-Fold Cross-validation accuracy')
#plt.xlabel('cycle')
#plt.ylabel('accuracy')
#plt.grid(True)
#plt.show()    
#############################验收##############################
train,labels=loadData("D:/QQfile/check_train.csv")
weights=LRGradDescent(train,labels,0.1,100)
predict=np.array(sigmoid(train*weights).transpose())[0]
predict=[0 if x<0.5 else 1 for x in predict]
correct=0
for i in range(len(predict)):
    if(predict[i]==labels[i]):
        correct+=1
print(correct/len(predict))

test,testLabels=loadData("D:/QQfile/check_test.csv")
predict=np.array(sigmoid(test*weights).transpose())[0]
predict=[0 if x<0.5 else 1 for x in predict]
print(predict)