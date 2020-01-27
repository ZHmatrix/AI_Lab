#无dropout 无全局池化 直接全连接
# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# 加载数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                       download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)
# cifar-10数据集的类别
classes = ('plane', 'car', 'bird', 'cat','deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 定义网络结构
class Net(nn.Module):

    # 构造函数，定义了网络的基本结构
    def __init__(self):
        # 使用Net的父类的初始化方法,即运行nn.Module的初始化函数
        super(Net, self).__init__()
        # 卷积层1:输入为图像(rgb3通道图像),输出为64张特征图,卷积核为3x3,padding为1
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        # 卷积层2:输入为64张特征图,输出为64张特征图,卷积核为3x3,padding为1
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        # 用global average pooling 代替最后的全连接层
        self.globalavgpool = nn.AvgPool2d(8, 8)
        # BatchNorm2d最常用于卷积网络中(防止梯度消失或爆炸)，设置的参数就是卷积的输出通道数
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        # drop 训练的常用技巧，训练时随机使一部分神经元失活，能够防止过拟合
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256*8*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    # 前向传播过程，定义了各个网络层之间的连接方式
    def forward(self, x):
        # input x->卷积层1->激活函数ReLU->正规化BatchNorm
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        #x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        #x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        # 用global average pooling 代替最后的全连接层
        #x = self.globalavgpool(x)
        #x = self.dropout50(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器使用Adam，初始学习率为0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 指定运行的设备，多卡机器可以通过cuda:后的数字指定哪张卡
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 将网络中的数据转换为对应设备的类型(GPU则需要转换成cuda数据类型，CPU不需要)
net.to(device)

# 训练计时
start = time.clock()
# 运行epoch数
EPOCH_NUM = 100
# 准确率作图
accuracy_range = []
for epoch in range(EPOCH_NUM):

    running_loss = 0.
    batch_size = 100
    # 获取索引和数据
    for i, data in enumerate(
            torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0), 0):
        # data包含数据和标签信息，分别赋值给inputs和labels
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清0，因为反向传播过程中梯度会累加上一次循环的梯度
        optimizer.zero_grad()
        # 将训练数据数据输入网络得到输出
        outputs = net(inputs)
        # 使用交叉熵损失函数计算输出和label的误差值
        loss = criterion(outputs, labels)
        # 反向传播计算梯度
        loss.backward()
        # 执行反向传播后，优化器更新参数(Adam SGD...)
        optimizer.step()
        
        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))
        
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    accuracy_range.append(100 * correct / total)
    
print('%d epoch Finished Training'%EPOCH_NUM)
end = time.clock()
print("Training Time: ",str(end-start))

torch.save(net, 'cifar10_origin.pkl')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

epoch_range = np.arange(1,EPOCH_NUM+1)  
accuracy_range1=accuracy_range.copy()
plt.title("epoch-accuracy") 
plt.xlabel("epoch") 
plt.ylabel("accuracy") 
plt.plot(epoch_range, accuracy_range) 
plt.show()