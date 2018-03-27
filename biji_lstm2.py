import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
time_step = 28
## 特征，可以理解是影响因素
n_hidden = 256
n_classes = 10

x = tf.placeholder("float", [None, time_step, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

## 切成行，切成time_step份，每一行[batch_size的样本 * n_input]的矩阵  n_input做列
## 横向切割，注意这不是一个二维
train_x = tf.unstack(x, time_step, 1)
print(train_x)

lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)

## 理解成time_step的时刻，每个时刻一个tensor
## 一个cell一个时刻，一个时刻一个输出
## 此时，输入X已经转化成输出 [batch_size的样本 ,n_hidden个特征]
## 每一个小cell的输入，是[batch_size * n_inputs]
outputs, states = rnn.static_rnn(lstm_cell, train_x, dtype=tf.float32)

## 提取最后一个输出，进入全连接层，得到[bacth_size,classes]的y_
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
        ## 这个时候进入模型,只需要将输入变换，预测也是这样
        batch_xs = batch_xs.reshape([batch_size, time_step, n_input])
        sess.run(optimizer,feed_dict={
            x:batch_xs,y:batch_ys
        })
        if i==9:
            print(sess.run(accuracy,feed_dict={
                x: batch_xs, y: batch_ys
            }))
