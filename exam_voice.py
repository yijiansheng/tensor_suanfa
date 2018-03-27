import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf  # 0.12
from sklearn.model_selection import train_test_split


voice_data = pd.read_csv('dataset/voice.csv')
voice_data = voice_data.values
voices = voice_data[:, :-1]
labels = voice_data[:, -1:]

## label转向量
labels_tmp = []
for label in labels:
    tmp = []
    if label[0] == 'male':
        tmp = [1.0, 0.0]
    else:  # 'female'
        tmp = [0.0, 1.0]
    labels_tmp.append(tmp)
labels = np.array(labels_tmp)

voices_tmp = []
lables_tmp = []
## [0 ,……,length]
index_shuf = [i for i in range(len(voices))]
random.shuffle(index_shuf)
## print(index_shuf)
for i in index_shuf:
    voices_tmp.append(voices[i])
    lables_tmp.append(labels[i])
voices = np.array(voices_tmp)
labels = np.array(lables_tmp)

print(voices.shape)
print(labels.shape)

train_x, test_x, train_y, test_y = train_test_split(voices, labels, test_size=0.1)
batch_size = 64
n_batch_size = len(train_x) // batch_size

X = tf.placeholder(dtype=tf.float32, shape=[None, voices.shape[-1]])  # 20
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])



def neural_network():
    ## 从正态分布中得到平均值,stddev标准差
    ## 一层
    w1 = tf.Variable(tf.random_normal([voices.shape[-1], 512], stddev=0.5))
    b1 = tf.Variable(tf.random_normal([512]))
    output = tf.matmul(X, w1) + b1

    ## 从512 到1024个特征
    w2 = tf.Variable(tf.random_normal([512, 1024], stddev=.5))
    b2 = tf.Variable(tf.random_normal([1024]))
    ## 非线性
    output = tf.nn.softmax(tf.matmul(output, w2) + b2)

    w3 = tf.Variable(tf.random_normal([1024, 2], stddev=.5))
    b3 = tf.Variable(tf.random_normal([2]))
    output = tf.nn.softmax(tf.matmul(output, w3) + b3)
    return output


# 训练神经网络
def train_neural_network():
    output = neural_network()
    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y)))
    ## 不要更新变量
    lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    var_list = [t for t in tf.trainable_variables()]
    train_step = opt.minimize(cost, var_list=var_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(40):
            ## assign是赋值
            sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))
            for batch_index in range(n_batch_size):
                voice_banch = train_x[batch_index * batch_size:(batch_index + 1) * (batch_size)]
                label_banch = train_y[batch_index * batch_size:(batch_index + 1) * (batch_size)]
                _, loss = sess.run([train_step, cost], feed_dict={X: voice_banch, Y: label_banch})
                print(epoch, batch_index, loss)

                # 准确率
        prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
        accuracy = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
        print("准确率", accuracy)

train_neural_network()