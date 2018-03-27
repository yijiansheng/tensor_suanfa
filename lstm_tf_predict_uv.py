import pandas as pd
import numpy as np
import tensorflow as tf

f=open('dataset/guangzhou_dongguan.csv')
df=pd.read_csv(f)
data=np.array(df['per_uv'])


data=data[0:len(data)-1]
#归一化
pv_data=(data-np.mean(data))/np.std(data)
##N行一个特征
pv_data=pv_data[:,np.newaxis]

time_step=10
batch_size=20
rnn_unit=5

## 特征数量
feature_num=1
## 输出向量数量
pred_num=1

##拿到train_x的总量，并且初始化
train_x,train_y=[],[]
for i in range(len(pv_data)-time_step-1):
    x=pv_data[i:i+time_step]
    y=pv_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())


X=tf.placeholder(tf.float32, [None,time_step,feature_num])
Y=tf.placeholder(tf.float32, [None,time_step,pred_num])

## 先将feature_num转成lstm识别的输出
weights={
    'in':tf.Variable(tf.random_normal([feature_num,rnn_unit])),
    'out':tf.Variable(tf.random_normal([rnn_unit,pred_num]))
}

biases={
    'in':tf.Variable(tf.constant(0.1)),
    'out':tf.Variable(tf.constant(0.1))
}

## 构造lstm网络，转化输入的X，套进方程
def init_lstm(batch_num):
    w_in = weights['in']
    b_in = biases['in']
    ## 只保留一个特征，其余变成行
    input=tf.reshape(X,[-1,feature_num])
    ## 保留行和w的列
    input_rnn = tf.matmul(input, w_in) + b_in
    ## reshape回来
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_num,dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    ## 输出是N*layer的矩阵
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    ## 计算模型
    w_out = weights['out']
    b_out = biases['out']
    ## 最后pred是N*1
    predict_y_result = tf.matmul(output, w_out) + b_out
    return predict_y_result




def train_weight():
    predict_y_result = init_lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(predict_y_result, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        start = 0
        end = start + batch_size
        while (end < len(train_x)):
            sess.run(train_op, feed_dict={X: train_x[start:end], Y: train_y[start:end]})
            ##训练样本有限，一次性跨度少一些
            start += batch_size
            end = start + batch_size

    print(sess.run(weights['out']))
    print(sess.run(biases['out']))

train_weight()