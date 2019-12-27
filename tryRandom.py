# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import random

from HelperClass2.NeuralNet_2_0 import *

train_data_name = ".\data\ch09.train.npz"
test_data_name = ".\data\ch09.test.npz"

def train(hp, folder):
    net = NeuralNet_2_0(hp, folder)
    net.train(dataReader, 50, True)
    print("Accuracy: ", net.Test(dataReader), "eta: ", hp.eta)
    trace = net.GetTrainingHistory()
    return trace

def try_hyperParameters(folder, n_hidden, batch_size, eta):
    hp = HyperParameters_2_0(1, n_hidden, 1, eta, 10000, batch_size, 0.001, NetType.Fitting, InitialMethod.Xavier)
    filename = str.format("{0}\\{1}_{2}_{3}.pkl", folder, n_hidden, batch_size, eta).replace('.', '', 1)
    file = Path(filename)
    if file.exists():
        return file, hp
    else:
        lh = train(hp, folder)
        lh.Dump(file)
        return file, hp

#  根据超参数来载入已经训练好的文件
def load_hyperParameters(folder, n_hidden, batch_size, eta):
    filename = str.format("{0}\\{1}_{2}_{3}.pkl", folder, n_hidden, batch_size, eta).replace('.', '', 1)
    file = Path(filename)
    return file

#  得到训练的最终的正确率
def load_accuracy(file):
    lh=TrainingHistory_2_0.Load(file)
    return lh.accuracy_val[-1]


def ShowResult2D(net, title):
    count = 21
    
    TX = np.linspace(0,1,count).reshape(count,1)
    TY = net.inference(TX)

    print("TX=",TX)
    print("Z1=",net.Z1)
    print("A1=",net.A1)
    print("Z=",net.Z2)

    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,np.zeros((count,1)),'.',c='black')
    p2,= plt.plot(TX,net.Z1[:,0],'.',c='r')
    p3,= plt.plot(TX,net.Z1[:,1],'.',c='g')
    plt.legend([p1,p2,p3], ["x","z1","z2"])
    plt.grid()
    plt.show()
    
    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,np.zeros((count,1)),'.',c='black')
    p2,= plt.plot(TX,net.Z1[:,0],'.',c='r')
    p3,= plt.plot(TX,net.A1[:,0],'x',c='r')
    plt.legend([p1,p2,p3], ["x","z1","a1"])
    plt.grid()
    plt.show()

    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,np.zeros((count,1)),'.',c='black')
    p2,= plt.plot(TX,net.Z1[:,1],'.',c='g')
    p3,= plt.plot(TX,net.A1[:,1],'x',c='g')
    plt.legend([p1,p2,p3], ["x","z2","a2"])
    plt.show()

    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,net.A1[:,0],'.',c='r')
    p2,= plt.plot(TX,net.A1[:,1],'.',c='g')
    p3,= plt.plot(TX,net.Z2[:,0],'x',c='blue')
    plt.legend([p1,p2,p3], ["a1","a2","z"])
    plt.show()
    
def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
    
if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    #   eta这样的参数希望的是对数均匀分布的，那么0.001~0.01,0.01~0.1,0.1~1的抽样概率是类似的
    #   以下是10^-3~10^0的对数均匀分布
    eta_list = np.logspace(-4, 0, base=10, num=100) 
    #   batch这样的参数，我不知道怎样的分布合适
    #   https://arxiv.org/abs/1804.07612 
    #   这篇文章说2~10比较合适
    batch_list = list(range(2, 10))
    #   n_hidden 这个参数按照网上的经验公式可以得到 h=sqrt(m+n)+a,m为输入层节点数目，n为输出层节点数目，a为之间的调节常数1~10
    ne_list = list(range(1, 11))
    
    n_input, n_output = 1, 1
    max_epoch = 10000
    eps = 0.001
    
    max_acc = 0
    best_eta = 0
    best_batch = 0
    best_ne = 0
    folder = "file"
    
    for KASE in range(40):
        eta = random.choice(eta_list)
        batch = random.choice(batch_list)
        ne = random.choice(ne_list)
        file, hp = try_hyperParameters(folder, ne, batch, eta)
        if max_acc < load_accuracy(file):
            max_acc = load_accuracy(file)
            best_eta = eta
            best_batch = batch
            best_ne = ne
        #endif
    #endfor
    
    hp = HyperParameters_2_0(n_input, best_ne, n_output, best_eta, max_epoch, best_batch, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet_2_0(hp, folder)
    net.train(dataReader, 50, True)
    net.ShowTrainingHistory()
    print("best_ne:",best_ne," best_batch:",best_batch," best_eta:", best_eta, "acc:", max_acc) 
    ShowResult(net, dataReader, hp.toString())