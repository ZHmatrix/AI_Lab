# -*- coding: utf-8 -*-
import string
import time
import math

from datetime import datetime

import numpy as np

import torch
from torch.autograd import Variable
from model import my_lstm
import torch.nn as nn
from gensim.models import word2vec

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
# 输入为数据，字典，以及预训练的embedding模型
def get_word_vect(data,word_dict,model):
    result=[]
    # 获取词典长度，加1是为了存放不在词典中出现的单词
    dict_len=len(word_dict)+1
    word_set=set(word_dict.keys())

    # 每个词的初始向量
    vector = [0] * 101

    # 对每一行操作
    for v in data:
        temp=[]
        temp.append(v[0])
        for s in [v[1],v[2]]:
            # 每个句子表示为二维矩阵，行数是词的个数，列数是词表的大小
            sentence=[]
            for word in s:
                # vector_copy=vector.copy()
                # if word in word_set:
                #     vector_copy[word_dict[word]]=1
                # else:
                #     vector_copy[-1]=1
                if word in model:
                    word_vec=np.append(model[word],0)
                    sentence.append([word_vec])
                else:
                    vector_copy=vector.copy()
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

    # 加载数据，进行切词，去标点
    train_data_raw = load_data('data/sts-train.txt')
    dev_data_raw = load_data('data/sts-dev.txt')
    # 获得词典
    word_dict = get_word_dict(train_data_raw, dev_data_raw)
    dict_len = 101
    print("The length of the word_list is " + str(dict_len))
    # 获取预训练好的模型
    model = word2vec.Word2Vec.load('word2vec_corpus/word2vec.model')
    train_data=get_word_vect(train_data_raw,word_dict,model)
    dev_data=get_word_vect(dev_data_raw,word_dict,model)

    print("Done preprocessing data, spent "+timeSince(prerocess_start))
    # ---------------------以上是数据处理部分----------------------------


    # ---------------------模型参数--------------------------
    # 输入向量的特征维度
    embed_size=dict_len
    batch_size=1
    # 隐藏层维度
    hidden_size=128
    # 单层 LSTM
    num_layers=3
    # dropout概率
    drop_out_prob=0
    # 初始化模型
    lstm = my_lstm(embed_size,batch_size,hidden_size,num_layers,drop_out_prob)
    lstm=lstm.to(device)
    # -----------以上是模型参数部分-----------------------

    # 学习率
    learning_rate = 0.1
    # 优化器（梯度更新）
    optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, lstm.parameters()), lr=learning_rate)
    # 损失函数
    criterion = nn.MSELoss()
    criterion=criterion.to(device)

    # 最好的一次损失
    best_loss = float('inf')

    # 迭代次数
    n_epoch = 10
    epoch = 1

    train_start_time=time.time()
    # 开始训练
    while epoch < n_epoch:
        print('Start Epoch {} Training...'.format(epoch))

        epoch_start_time=time.time()
        # 训练误差
        train_loss = []

        for idx, data in enumerate(train_data):
            # 获取数据
            original_score = torch.tensor(data[0], dtype=torch.float32).to(device)
            sentence_tensor1=torch.tensor(data[1],dtype=torch.float32).to(device)
            sentence_tensor2=torch.tensor(data[2],dtype=torch.float32).to(device)

            # 清除梯度
            optimizer.zero_grad()
            # 输出
            output = lstm(sentence_tensor1, sentence_tensor2)

            original_score = Variable(original_score)
            original_score.to(device)

            # 误差反向传播
            loss = criterion(output, original_score)
            loss.backward()

            # 更新梯度
            optimizer.step()
            train_loss.append(loss.data.cpu())
        print("Train this epoch spent: "+timeSince(epoch_start_time))
        print('Epoch %d train mean loss: %f' %(epoch,sum(train_loss)/len(train_loss)))

        valid_loss = []
        valid_start_time=time.time()
        for idx, data in enumerate(dev_data):
            original_score = torch.tensor(data[0], dtype=torch.float).to(device)
            sentence_tensor1 = torch.tensor(data[1], dtype=torch.float).to(device)
            sentence_tensor2 = torch.tensor(data[2], dtype=torch.float).to(device)

            output = lstm(sentence_tensor1, sentence_tensor2)

            original_score = Variable(original_score)
            original_score.to(device)

            loss = criterion(output, original_score)
            valid_loss.append(loss.data.cpu())
        print("Eval this epoch spent: "+timeSince(valid_start_time))
        print('Epoch %d valid mean loss: %f' % (epoch, sum(valid_loss) / len(valid_loss)))

        # 查看与最好的loss的差距
        if np.mean(valid_loss) < best_loss:
            best_loss = np.mean(valid_loss)
            torch.save(lstm.state_dict(), "lstm_params.pkl")
        print("Epoch %d spent %s" %(epoch,timeSince(epoch_start_time)))
        epoch += 1

    print("Train spent "+timeSince(train_start_time))

