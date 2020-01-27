# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:52:12 2019

@author: 仲逊
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

def normalize(num_mat):
    samples_size=num_mat.shape[0]
    ans=[]
    for i in range(samples_size):
        ans.append(num_mat[i]/np.sum(num_mat[i]))
    return np.array(ans)
    
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
            cos=np.dot(X_test[i],X_train[j])/(np.linalg.norm(X_test[i])*(np.linalg.norm(X_train[j])))
            dists[i][j]=1-cos
    return dists
    
def KNN_regression(X_train,y_train,X_valid,y_valid,k,p):
    """
    input：X_train 训练集 numpy矩阵  y_train 训练标签 numpy矩阵
           X_test 测试集 numpy矩阵  y_test 测试集标签 numpy矩阵
           k KNN的参数 整数
           p Lp范数 整数  
    return：y_predict 预测结果 numpy矩阵  
    """
    #dists=compute_cos_distances(X_train, X_valid)
    dists=compute_distances(X_train, X_valid, p)
    sorted_dists=np.sort(dists)                  #排好序的距离
    sorted_mat=y_train[np.argsort(dists)] #根据距离排好序的训练集标签

    y_predict=[]
    for m in range(sorted_mat.shape[0]):
        sorted_array=sorted_mat[m]
        #各个情感的概率，用列表存储
        p=[]
        for j in range(sorted_array.shape[1]):
            #使用距离倒数对最近K个样例的情感指标进行加权
            p_of_motion=0        
            for i in range(k):
                p_of_motion+=sorted_array[i][j]/sorted_dists[m][i]
            p.append(p_of_motion)
        y_predict.append(p.copy())
    #归一化处理
    y_predict=normalize(np.array(y_predict))
    #计算相关系数的平均值
#    sum_corrcoef=0
#    for i in range(y_valid.shape[1]):
#        sum_corrcoef+=np.corrcoef(y_valid[:,i],y_predict[:,i])
#    print("corrcoef:",(sum_corrcoef/y_valid.shape[1])[0][1])
    
    return y_predict
 
    
##########################加载数据############################
#train_labels=pd.read_csv("D:/AI Lab/lab1/lab1_data/regression_dataset/train_set.csv")
#validation_labels=pd.read_csv("D:/AI Lab/lab1/lab1_data/regression_dataset/validation_set.csv")
#train=train_labels.pop("Words (split by space)")
#validation=validation_labels.pop("Words (split by space)")
#test_labels=pd.read_csv("D:/AI Lab/lab1/lab1_data/regression_dataset/test_set.csv")
#test=test_labels.pop("Words (split by space)")
#test_labels=pd.DataFrame(np.zeros((len(test.values),6)))

####################测试验证集的相关系数#######################
#使用训练集和验证集的单词建立词典
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].split(" "))
#for i in range(len(validation)):
#    senten_list.append(validation.loc[i].split(" "))
#words_list=get_words_list(senten_list)
##获取训练集的tfidf矩阵
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].split(" "))
#train=get_tfidf(senten_list,words_list)
##获取验证集的tfidf矩阵
#senten_list=[]
#for i in range(len(validation)):
#    senten_list.append(validation.loc[i].split(" "))
#validation=get_tfidf(senten_list,words_list)
#
#for k in range(1,20):
#    ans=KNN_regression(train.values,train_labels.values,validation.values,validation_labels.values,k,1)

#####################预测并写入文件########################
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].split(" "))
#for i in range(len(test)):
#    senten_list.append(test.loc[i].split(" "))
#words_list=get_words_list(senten_list)
##获取训练集的tfidf矩阵
#senten_list=[]
#for i in range(len(train)):
#    senten_list.append(train.loc[i].split(" "))
#train=get_tfidf(senten_list,words_list)
#
#senten_list=[]
#for i in range(len(test)):
#    senten_list.append(test.loc[i].split(" "))
#test=get_tfidf(senten_list,words_list)
#
#ans=KNN_regression(train.values,train_labels.values,test.values,test_labels.values,5,1)
#df=pd.DataFrame(ans,columns=["anger","disgust","fear","joy","sad","surprise"])
#df.insert(0,'textid',range(1,len(df.values)+1))
#df.to_csv("16327143_zhongxun_KNN_regression.csv",index=False)

#######################验收###############################
train_label=pd.read_csv("C:/Users/acer/Desktop/regression_simple_test.csv")
train=train_label.pop("Words (split by space)")

senten_list=[]
for i in range(len(train)):
    senten_list.append(train.loc[i].split(" "))
words_list=get_words_list(senten_list)

test=train.drop([0,1,2,3,4,5])
train=train.drop(index=6)
train_label=train_label.drop(index=6)
test_label=pd.DataFrame(np.array([[0,0,0,0,0,0]]),columns=["anger","disgust","fear","joy","sad","surprise"])

senten_list=[]
for i in range(len(train)):
    senten_list.append(train.loc[i].split(" "))

train=get_tfidf(senten_list,words_list)
senten_list=[]
senten_list.append(test.loc[6].split(" "))

test=get_tfidf(senten_list,words_list)
train_label=train_label.applymap(lambda v:float(v))

ans=KNN_regression(train.values,train_label.values,test.values,test_label.values,3,2)
print(ans)