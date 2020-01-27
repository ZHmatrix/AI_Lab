from torch import nn
from torch.autograd import Variable
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LSTM结构
class LSTMEncoder(nn.Module):
    # 初始化LSTM模型
    def __init__(self, embed_size,batch_size,hidden_size,num_layers,drop_out_prob):
        super(LSTMEncoder, self).__init__()
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = drop_out_prob
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout)
    # 初始化隐藏层状态以及细胞状态
    def initHiddenCell(self):
        rand_hidden =Variable(torch.randn( self.num_layers, self.batch_size, self.hidden_size).to(device))
        rand_cell = Variable(torch.randn( self.num_layers, self.batch_size, self.hidden_size).to(device))
        return rand_hidden, rand_cell
    # 前向传播
    def forward(self, input, hidden, cell):
        input = input.view(1, 1, -1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

# 使用LSTM对句子进行训练
class my_lstm(nn.Module):
    def __init__(self,embed_size,batch_size,hidden_size,num_layers,drop_out_prob):
        # 初始化
        super(my_lstm, self).__init__()
        self.encoder = LSTMEncoder(embed_size,batch_size,hidden_size,num_layers,drop_out_prob)

    # 前向传播
    def forward(self, s1, s2):
        # 初始化隐层状态以及细胞状态
        h1, c1 = self.encoder.initHiddenCell()
        h2, c2 = self.encoder.initHiddenCell()
        # 一个一个把单词输入进去
        o1=0
        o2=0
        for i in range(len(s1)):
            o1, h1, c1 = self.encoder(s1[i], h1, c1)
        for j in range(len(s2)):
            o2, h2, c2 = self.encoder(s2[j], h2, c2)
        # 计算exp(-||o1_o2||_1)后还要乘以5映射到0-5
        output= 5*torch.exp(-torch.norm(o1-o2,p=1))
        return output