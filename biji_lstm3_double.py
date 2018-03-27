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
## 时间步，决定lstm有几层
n_steps = 28
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

## 特征层有两组序列
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

train_x = tf.unstack(x, n_steps, 1)
## 前向
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
## 后向
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, train_x ,dtype=tf.float32)

## 取最后一个时间步的输出
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

## session执行
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        ## 这个时候进入模型,只需要将输入变换，预测也是这样
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
        sess.run(optimizer,feed_dict={
            x:batch_xs,y:batch_ys
        })
        if i==99:
            print(sess.run(accuracy,feed_dict={
                x: batch_xs, y: batch_ys
            }))