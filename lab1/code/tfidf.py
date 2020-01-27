# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:48:48 2019

@author: 16327143仲逊
"""
import numpy as np
import pandas as pd
def get_words_list(senten_list):
    '''
    input：senten_list 句子列表 每个句子是由word构成的列表
    return：words_list，由不重复单词组成的列表，按出现顺序排列
    '''
    words_list=[]                   #有序词汇列表
    vocabulary=set([])              #无序词汇表
    
    for sentence in senten_list:
        for word in sentence:
            if word not in vocabulary:
                words_list.append(word)
                vocabulary.add(word)   
    return words_list.copy()

def get_tfidf(senten_list,words_list):
    '''
    input：senten_list 句子列表 每个句子是由word构成的列表
           words_list 由不重复单词组成的列表
           根据给定词典和文章建立tfidf矩阵
    return：tfidf矩阵，类型为pandas的dataframe
    '''
    vocabulary=set(words_list)
    samples_num=len(senten_list)    #样本数
    idf=[]                          #逆向文本频率
    #生成维度为sample数×词汇表长度的0矩阵,
    tfidf=pd.DataFrame(np.zeros((samples_num,len(words_list)),dtype=np.int),columns=words_list)
    #现在的矩阵元素表示对应单词在该sample中出现了多少次
    for i in range(len(senten_list)):
        for word in senten_list[i]:
            if word in vocabulary:
                tfidf[word][i]+=1

    #计算每个单词对应的idf，存在列表中
    for word in words_list:
        #idf.append(np.log10(samples_num/len([1 for x in tfidf[word].values if x>0])))
        idf.append(np.log((samples_num+1)/(len([1 for x in tfidf[word].values if x>0])+1)))
    #归一化，现在的矩阵元素表示对应的词语出现次数归一化后的频率，即tf
    for i in range(samples_num):
        tfidf.loc[i]/=tfidf.loc[i].values.sum()
    #将矩阵每项×对应的idf得到tf-idf矩阵
    for i in range(len(words_list)):
        tfidf[words_list[i]]*=idf[i]

    return tfidf
        
if __name__ == '__main__':
    senten_list=[]                      #句子列表
    semeval=open("D:/AI Lab/lab1/lab1_data/semeval.txt","r")
    for line in semeval.readlines():
        #去除换行符后，以"\t"为间隔划分
        tem=line.strip().split("\t")
        #从句子中建立词汇表
        sentence=tem[2].split(" ")
        senten_list.append(sentence.copy())
    semeval.close()
    #建立词典
    words_list=get_words_list(senten_list)
    #获取tfidf矩阵
    tfidf=get_tfidf(senten_list,words_list)
    np.savetxt("16327143_zhongxun_TFIDF.txt",np.mat(tfidf.values),fmt="%.6f")