# -*- coding: utf-8 -*-
import string
import time
import math

from datetime import datetime

import numpy as np

import torch
from torch.autograd import Variable
from model import Siamese_lstm
import torch.nn as nn


# 输入：文件名
# 输出：二维矩阵，每一行代表一个训练样本里的两个句子以及相似度
def load_data(filename):
    file=open(filename,'r',encoding='utf-8')
    result=[]
    for line in file:
        # 取第4列到第6列
        # 分别是 相似度 句子1 句子2
        temp=[]
        li = line.split('\t')[4:7]
        temp.append(float(li[0]))
        for sentence in [li[1],li[2]]:
            for punc in string.punctuation:
                sentence=sentence.replace(punc,' ')
            sentence=sentence.replace('  ',' ')
            sentence=sentence.strip().split(' ')

            temp.append(sentence)
        result.append(temp)

    return result

# 构建词典
# 输入是train和dev的数据
def get_word_dict(train_data,dev_data):
    word_set=set()
    dict={}
    # cnt储存字的位置
    cnt=0
    for data in [train_data,dev_data]:
        for v in data:
            # 句子1和句子2
            for s in [v[1],v[2]]:
                for word in s:
                    if word not in word_set:
                        word_set.add(word)
                        dict[word]=cnt
                        cnt+=1
    return dict

# 获得表示成向量的词语
def get_word_vect(data,word_dict):
    result=[]
    # 获取词典长度，加1是为了存放不在词典中出现的单词
    dict_len=len(word_dict)+1
    word_set=set(word_dict.keys())

    # 每个词的初始向量
    vector = [0] * dict_len

    # 对每一行操作
    for v in data:
        temp=[]
        temp.append(v[0])
        for s in [v[1],v[2]]:
            # 每个句子表示为二维矩阵，行数是词的个数，列数是词表的大小
            sentence=[]
            for word in s:
                vector_copy=vector.copy()
                if word in word_set:
                    vector_copy[word_dict[word]]=1
                else:
                    vector_copy[-1]=1
                sentence.append([vector_copy])
            temp.append(sentence)
        result.append(temp)
        # print(temp)
    return result

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__=='__main__':
    # ------------------数据处理---------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Now using: "+str(device))

    # 开始时间
    prerocess_start = time.time()

    train_data_raw=load_data('data/sts-train.txt')[:100]
    dev_data_raw=load_data('data/sts-dev.txt')[:100]
    test_data_raw=load_data('data/sts-test.txt')

    word_dict=get_word_dict(train_data_raw,dev_data_raw)
    dict_len=len(word_dict)+1
    print("The length of the word_list is "+str(dict_len))
    train_data=get_word_vect(train_data_raw,word_dict)
    dev_data=get_word_vect(dev_data_raw,word_dict)
    test_data=get_word_vect(test_data_raw,word_dict)

    print("Done preprocessing data, spent "+timeSince(prerocess_start))
    # ---------------------以上是数据处理部分----------------------------


    # ---------------------模型参数--------------------------
    # 输入向量的特征维度
    embed_size=dict_len
    batch_size=1
    # 隐藏层维度
    hidden_size=128
    # 单层 LSTM
    num_layers=1
    # dropout概率
    drop_out_prob=0
    # 初始化模型
    siamese = Siamese_lstm(embed_size,batch_size,hidden_size,num_layers,drop_out_prob)
    siamese=siamese.to(device)
    # -----------以上是模型参数部分-----------------------

    siamese.load_state_dict(torch.load('lstm_params.pkl'))

    all_score=[]

    for idx,data in enumerate(test_data):
        # 获取数据
        sentence_tensor1 = torch.tensor(data[1], dtype=torch.float32).to(device)
        sentence_tensor2 = torch.tensor(data[2], dtype=torch.float32).to(device)

        # 输出
        output = siamese(sentence_tensor1, sentence_tensor2)

        all_score.append(output.data.cpu().item())

    print(all_score)
    csv_to_write=open('predict_test_data.csv','w',encoding='utf-8')
    for score in all_score:
        csv_to_write.write(str(score))
        csv_to_write.write('\n')