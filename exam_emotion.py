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


## 构建词典
def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with open(train_file, buffering=10000, encoding='latin-1') as f:
        try:
            count_word = {}  # 统计单词出现次数
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words = word_tokenize(tweet.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1

            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            for word in count_word:
                if count_word[word] < 100000 and count_word[word] > 10:  # 过滤掉一些词
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex


# wordsDic = create_lexicon('dataset/training.csv')
#
# with open('dataset/lexcion.pickle', 'wb') as f:
#     pickle.dump(wordsDic, f)

f = open('dataset/lexcion.pickle', 'rb')
wordsDic = pickle.load(f)
f.close()



def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()


def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines


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

# def f():
#     a = []
#     b = []
#     a.append(1)
#     b.append(1)
#     try:
#         b.append(eval("123", 213214))
#         a.append(2)
#     except :
#         print("false")
#     return a,b
# a,b = f()
# print(a)
# print(b)

test_x, test_y = get_test_dataset('dataset/tesing.csv')
train_x, train_y = get_train_dataset('dataset/training.csv')



## 定义层的神经元
n_input_layer = len(wordsDic)  # 输入层
n_layer_1 = 2000
n_layer_2 = 2000
n_output_layer = 3



##　构建NET
def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output



X = tf.placeholder('float',[None,len(train_x[0])])
Y = tf.placeholder('float',[None,len(train_y[0])])
batch_size = 50



##  训练
def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_func)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
            epoch_loss += c
            i += batch_size

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: list(test_x), Y: list(test_y)}))
        saver.save(session,'tmp/emotion.ckpt')
        print('save finish')

##　train_neural_network(X,Y)





## 预测 prediction,重新开一个session
X_Input = tf.placeholder("float",[None,len(train_x[0])])
input_x = []
def prediction(input_text):
    predict = neural_network(X_Input)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, 'tmp/emotion.ckpt')
        print('load finish')
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(input_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(wordsDic))
        for word in words:
            if word in wordsDic:
                features[wordsDic.index(word)] = 1
        print(features)
        input_x.append(list(features))
        result = session.run(tf.argmax(predict.eval(feed_dict={X_Input: input_x}), 1))
        print(predict.eval(feed_dict={X_Input: input_x}))
        print(result)

prediction("hello hello")