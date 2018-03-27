import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
import os
import random
import tensorflow as tf

f = open('dataset/lexcion.pickle', 'rb')
wordsDic = pickle.load(f)
f.close()

def get_test_dataset(test_file):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(wordsDic))
            for word in words:
                if word in wordsDic:
                    features[wordsDic.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y




def get_train_dataset(train_file):
    with open(train_file, encoding='latin-1') as f:
        train_x = []
        train_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(wordsDic))
            for word in words:
                if word in wordsDic:
                    features[wordsDic.index(word)] = 1

            try:
                train_y.append(eval(label))
                train_x.append(list(features))
            except:
                continue
    return train_x, train_y



test_x, test_y = get_test_dataset('dataset/tesing.csv')
train_x, train_y = get_train_dataset('dataset/training.csv')


##
input_size = len(wordsDic)
num_classes = 3
## 一条是一个input_size
X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)
batch_size = 90


## nlp的卷积
def neural_network():
    # embedding layer
    ## 分布式表征
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        ## 降低一个维度
        embedding_size = 128
        ##　len(wordsDic)  *   embedding
        ## 总的特征也就这么多
        W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
        ## 前面这个是数据集，后面是索引ids
        print(X)
        print(W)
        ## 多少行batch  * 每行 词vectors 固定长度  * 每一个词的embedding的表达,这是一个大img
        ## X_input
        embedded_chars = tf.nn.embedding_lookup(W, X)

        print(embedded_chars)
        ##　增加一个维度　，　改变输入
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        print(embedded_chars_expanded)
    # convolution + maxpool layer
    ## 卷积核个数,出来多少张
    num_filters = 128
    ## 看成一个整体
    filter_sizes = [3, 4, 5]
    ## 保存每次卷积的结果
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        ## filter_size个字作为一个整体考虑
        ## filter_size*embedding_size 一个字有embedding的表达
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            ## 卷积层W 将vector想成img 前两个是输入，第三个是输入通道，最后是输出数量
            ## 基础图像形状  filter_size * embedding, 1 , 输出多少张
            filter_shape = [filter_size,
                            embedding_size,
                            1,
                            num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            print("x_input")
            print(embedded_chars_expanded)
            print("conv_W")
            print(W)
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(
                ## input_x
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID")
            print("conv_result")
            print(conv)
            ## 激活函数
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            ## 池化 input_size 是一句话的序列总长度 ,len(words)
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID')
            print("pool")
            print(pooled)
            ## 128张 num_filters
            ##　batch * 1* 1 *num_filters
            pooled_outputs.append(pooled)
    ##　总共的输出图片数
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    ## 展开
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
        print("drop_input")
        print(h_drop)
        # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)
    return output




## 开始训练
def train_neural_network():
    output = neural_network()
    optimizer = tf.train.AdamOptimizer(0.01)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    predictions = tf.argmax(output, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))



    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        i = 0
        print(len(train_x))
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            train_opti = session.run(train_op, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
            i += batch_size
            print(i)

        print("train_finish")
        accu = session.run(accuracy, feed_dict={X: test_x[0:90], Y: test_y[0:90], dropout_keep_prob: 1.0})
        print(accu)

##　大量提高了正确率 49%
train_neural_network()