# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:52:17 2019

@author: 16327143仲逊
"""

import numpy as np
import pandas as pd
from tfidf import *
def compute_distances(X_train, X_test, p):
    """
    input：X_train 训练集 numpy矩阵
           X_test 测试集 numpy矩阵
           p Lp范数 整数 
           根据给定训练集和测试集，计算距离矩阵
    return：距离矩阵，mat[i][j]表示第i个验证样本到第j个训练样本的距离
    """
    test_num=X_test.shape[0]
    dists=[]
    for i in range(test_num):
        dist=np.power(np.sum(np.power(np.abs(X_train-X_test[i]),p),axis=1),1/p)
        dists.append(dist)
        
    return np.array(dists)

def compute_cos_distances(X_train, X_test):
    """
    input：X_train 训练集 numpy矩阵
           X_test 测试集 numpy矩阵
           根据给定训练集和测试集，计算余弦距离矩阵
    return：距离矩阵，mat[i][j]表示第i个验证样本到第j个训练样本的距离
    """
    test_num=X_test.shape[0]
    train_num=X_train.shape[0]
    dists=np.zeros((test_num,train_num))
    for i in range(test_num):
        for j in range(train_num):
            #cos=
            dists[i][j]=1-np.dot(X_test[i],X_train[j])/(np.linalg.norm(X_test[i])*(np.linalg.norm(X_train[j])))
    return dists

def KNN_classification(X_train,y_train,X_test,y_test,k,p):
    """
    input：X_train 训练集 numpy矩阵  y_train 训练标签 numpy矩阵
           X_test 测试集 numpy矩阵  y_test 测试集标签 numpy矩阵
           k KNN的参数 整数
           p Lp范数 整数  
    return：y_predict 预测结果 numpy矩阵  
    """
    dists=compute_distances(X_train, X_test, p)
#    dists=compute_cos_distances(X_train, X_test)
    sorted_mat=y_train[np.argsort(dists)]
    y_predict=[]
    for sorted_array in sorted_mat:
        y_predict.append(np.argmax(np.bincount(sorted_array[:k])))
    right_pred=0
    for i in range(len(y_predict)):
        if y_predict[i]==y_test[i]:
            right_pred+=1
#    print("accuracy = ",right_pred/len(y_predict))
    
    return np.array(y_predict)
    
motion2num={"joy":1,
            "sad":2,
            "surprise":3,
            "fear":4,
            "anger":5,
            "disgust":6
            }
num2motion={1:"joy",
            2:"sad",
            3:"surprise",
            4:"fear",
            5:"anger",
            6:"disgust"
            }  

##########################加载数据############################
#train=pd.read_csv("D:/AI Lab/lab1/lab1_data/classification_dataset/train_set.csv")
#validation=pd.read_csv("D:/AI Lab/lab1/lab1_data/classification_dataset/validation_set.csv")
#train_labels=train.pop("label")
#validation_labels = validation.pop("label")
#test_labels=pd.read_csv("D:/AI Lab/lab1/lab1_data/classification_dataset/test_set.csv")
#test=test_labels.pop("Words (split by space)")
#test_labels=pd.DataFrame(np.zeros((len(test.values)),dtype=np.int))

####################测试验证集的准确率########################
#for i in range(len(train_labels)):
#    train_labels.loc[i]=motion2num[train_labels.loc[i]]
#for i in range(len(validation_labels)):
#    validation_labels.loc[i]=motion2num[validation_labels.loc[i]]
#    
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].values[0].split(" "))
#for i in range(len(validation)):
#    senten_list.append(validation.loc[i].values[0].split(" "))
#words_list=get_words_list(senten_list)
#
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].values[0].split(" "))
#train=get_tfidf(senten_list,words_list)
#
#senten_list=[]
#for i in range(len(validation)):
#    senten_list.append(validation.loc[i].values[0].split(" "))
#validation=get_tfidf(senten_list,words_list)
#
#for k in range(1,20):
#    KNN_classification(train.values,train_labels.values,validation.values,validation_labels.values,k,2)
    
#######################预测并写入文件#######################
#for i in range(len(train_labels)):
#    train_labels.loc[i]=motion2num[train_labels.loc[i]]
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].values[0].split(" "))
#for i in range(len(test)):
#    senten_list.append(test.loc[i].split(" "))
#words_list=get_words_list(senten_list)
#
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].values[0].split(" "))
#train=get_tfidf(senten_list,words_list)
#
#senten_list=[]
#for i in range(len(test)):
#    senten_list.append(test.loc[i].split(" "))
#test=get_tfidf(senten_list,words_list)
#
#ans=KNN_classification(train.values,train_labels.values,test.values,test_labels.values,8,1)
#y_pridict=[]
#for i in range(0,len(ans)):
#    y_pridict.append(num2motion[ans[i]])
#df=pd.DataFrame(np.array(y_pridict),columns=["label"])
#df.insert(0,'textid',range(1,len(df.values)+1))
#df.to_csv("16327143_zhongxun_KNN_classification.csv",index=False)

########################验收#############################
train=pd.read_csv("C:/Users/acer/Desktop/classification_simple_test.csv")
test=train.drop([0,1,2,3,4,5])
train=train.drop(index=6)
train_labels=train.pop("label")
test_labels=test.pop("label")

for i in range(len(train_labels)):
    train_labels.loc[i]=motion2num[train_labels.loc[i]]

senten_list=[]
for i in range(len(train)):
    senten_list.append(train.loc[i].values[0].split(" "))

senten_list.append(test.loc[6].values[0].split(" "))
words_list=get_words_list(senten_list)

senten_list=[]
for i in range(len(train)):
    senten_list.append(train.loc[i].values[0].split(" "))
train=get_tfidf(senten_list,words_list)

senten_list=[test.loc[6,"Words (split by space)"].split(" ")]
    
test=get_tfidf(senten_list,words_list)
ans=KNN_classification(train.values,train_labels.values,test.values,test_labels.values,1,2)
print(num2motion[ans[0]])