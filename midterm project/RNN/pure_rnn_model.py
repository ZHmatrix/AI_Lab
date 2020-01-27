from torch import nn
from torch.autograd import Variable
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class rnnEncoder(nn.Module):
    # 初始化LSTM模型
    def __init__(self, embed_size,batch_size,hidden_size,num_layers,drop_out_prob):
        super(rnnEncoder, self).__init__()
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = drop_out_prob
        self.rnn=nn.RNN(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,dropout=self.dropout);

    def initHiddenCell(self):
        rand_hidden =Variable(torch.randn( self.num_layers, self.batch_size, self.hidden_size).to(device))
        return rand_hidden

    def forward(self, input, hidden):
        input = input.view(1, 1, -1)
        output, hidden = self.rnn(input,hidden)
        return output, hidden


class my_rnn(nn.Module):
    def __init__(self,embed_size,batch_size,hidden_size,num_layers,drop_out_prob):
        # 初始化
        super(my_rnn, self).__init__()
        self.encoder = rnnEncoder(embed_size,batch_size,hidden_size,num_layers,drop_out_prob)

    def forward(self, s1, s2):
        # 初始化隐层状态以及细胞状态
        h1 = self.encoder.initHiddenCell()
        h2 = self.encoder.initHiddenCell()

        # 一个一个把单词输入进去
        o1=0
        o2=0
        for i in range(len(s1)):
            o1, h1 = self.encoder(s1[i], h1)

        for j in range(len(s2)):
            o2, h2 = self.encoder(s2[j], h2)

        # 计算exp(-||o1_o2||_1)后还要乘以5映射到0-5
        output= 5*torch.exp(-torch.norm(o1-o2,p=1))

        return output