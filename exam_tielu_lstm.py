import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

f=open('dataset/tieludata.csv',encoding='utf8')
df=pd.read_csv(f)
data=np.array(df['value'])
normalized_data = (data - np.mean(data)) / np.std(data)
# plt.figure()
# plt.plot(data)
# plt.show()
## 142行 一维
print(normalized_data.shape)

## 序列长度
seq_size = 3
train_x = []
train_y = []
for i in range(len(normalized_data) - seq_size - 1):
    ## 训练数据
    train_x.append(np.expand_dims(normalized_data[i: i + seq_size], axis=1).tolist())
    train_y.append(normalized_data[i + 1: i + seq_size + 1].tolist())

print(train_x[0])
print(train_x[1])
print(train_x[2])
print(train_y[0])

##注意train_x和train_y train_x 是当前往后推3个 train_y 是先推一个后推三个
## trainx[1]和trainy[0]是一样的，但trainx多一维


## 因为一个数字就代表了一个量，所以input_dim
## 先把batch 想成1，进来X3个数，预测Y3个数
input_dim = 1
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])
Y = tf.placeholder(tf.float32, [None, seq_size])

## 一个cell产生的输出
hidden_layer_size = 6



def lstmnet(hidden_layer_size=6):
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')
    ## 一个cell里面，有hidden_layer_size个特征
    cell = rnn.BasicLSTMCell(hidden_layer_size)
    ## cell hidden=6
    ## X batch * 3 * 1
    ## outputs 改最后一维，一个步长一个结果  batch * 3 * hidden
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    ## 保留了每一步了
    output = tf.reshape(outputs, [-1, hidden_layer_size])
    pred = tf.matmul(output, W) + b
    return pred


## 训练过程
def train_rnn():
    out = lstmnet()
    input_y = tf.reshape(Y, [-1])
    loss = tf.reduce_mean(tf.square(out - input_y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[0:3], Y: train_y[0:3]})
            if step % 10 == 0:
                ## view_out , view_Y ,view_inputy = sess.run([out, Y, input_y], feed_dict={X: train_x[0:3], Y: train_y[0:3]})
                print(step, loss_)
                ## print(view_out,view_Y,view_inputy)
        ## print("保存模型: ", saver.save(sess, 'ass.model'))
        print("保存模型: ", saver.save(sess, 'tmp/tielu.model'))

## train_rnn()


def prediction():

    out = lstmnet()
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        saver.restore(sess, 'tmp/tielu.model')
        prev_seq = train_x[-1]
        predict = []
        for i in range(12):
            ## 预测输出
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            ## 重置下一个输入
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalized_data) + len(predict))), np.append(normalized_data, predict), color='r')
        plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        plt.show()
prediction()