import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

## 注意一点，lstm是以时刻为单位的，每一个时刻输入一行，这一行是28个单元
n_input = 28
## 时序长度，每做一次预测，需要先输入28行
time_step = 28 # timesteps
## 可以理解是神经元数量，特征数量
n_hidden = 128
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, time_step, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# def RNN(x, weights, biases):
#
#     ## 要做叉积操作,就必须二维
#     ## 变换完了之后，换回三维
#     ##  X = tf.reshape(X, [-1, n_inputs])
#     x = tf.unstack(x, time_step, 1)
#
#     ## 后面的参数是遗忘度
#     ## 前面的hidden是一个cell中，“神经元”的个数
#     ## cell的个数？
#
#     ## 任意时刻，一个cell内部，会产生两个内部状态，ct和ht，当这个参数为true，就是分开记录
#     lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
#     ## 输出对应于每一个time_step
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']

## 要做叉积操作,就必须二维
##  X = tf.reshape(X, [-1, n_inputs])
##  128batch*28timestep  *   28inputs
## 成为这么多行的输入
## 这个输入的意义是：batch_size,第一个cell的时间步长，特征数量
train_x = tf.unstack(x, time_step, 1)

## 后面的参数是遗忘度
## 前面的hidden是一个cell中，“神经元”的个数
## cell的个数？


## 任意时刻，一个cell内部，会产生两个内部状态，ct和ht，当这个参数为true，就是分开记录
##告知一个cell，需要多少个特征，即多少个神经元，它可以自动匹配X
## 将输入按照  时间线铺开的概念
## 这个模型有一些列的cell，时间不同步的cell
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
## 输出对应于每一个time_step
## 在这里面已经进行了s对X的叉积计算
outputs, states = rnn.static_rnn(lstm_cell, train_x, dtype=tf.float32)
## 全连接
## 128*128  x   128*10
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        ##　先将输入降低一个维度
        batch_xs = batch_xs.reshape([batch_size, time_step, n_input])
        ##训练一百次
        sess.run(optimizer,feed_dict={
            x:batch_xs,y:batch_ys
        })
        if i==9:
            print(sess.run(accuracy,feed_dict={
                x: batch_xs, y: batch_ys
            }))
            output_result = sess.run(outputs,feed_dict={
                x: batch_xs, y: batch_ys
            })
            print(len(output_result))
            ## 28  *  128(batch个数) *128 (cell的特征数量)
            #print(output_result[-1].shape)
            train_x_result = sess.run(train_x, feed_dict={
                x: batch_xs, y: batch_ys
            })
            ## 28*128*28
            print(train_x_result[0].shape)

