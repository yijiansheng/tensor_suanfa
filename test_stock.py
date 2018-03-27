import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
f=open('dataset/dataset_1.csv')
df=pd.read_csv(f)     #读入股票数据
##price
data=np.array(df['price'])
#以折线图展示data

normalize_data=(data-np.mean(data))/np.std(data)  #标准化
## 从一维到二维， 原N列，变成N行1列
normalize_data=normalize_data[:,np.newaxis]

time_step=20      #时间步
rnn_unit=10       #hidden layer units
batch_size=60     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集

for i in range(len(normalize_data)-time_step-1):
    # i -- i+time_step
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    ## 6090行，每一行
    train_x.append(x.tolist())

    train_y.append(y.tolist())

##print(len(train_x[6089]))