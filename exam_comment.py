import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import codecs

pos_file = 'dataset/pos.txt'
neg_file = 'dataset/neg.txt'



# 创建词汇表
def create_lexicon(pos_file, neg_file):
    allWordArr = []
    def process_file(f):
        with open(pos_file,'r') as f:
            wordArr = []
            lines = f.readlines()
            for line in lines:
                ##　一行分词一个list
                words = word_tokenize(line.lower())
                wordArr += words
            return wordArr

    allWordArr += process_file(pos_file)
    allWordArr += process_file(neg_file)
    lemmatizer = WordNetLemmatizer()
    # 词根还原
    allWordArr = [lemmatizer.lemmatize(word) for word in allWordArr]
    word_count = Counter(allWordArr)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    wordsDic = []
    for word in word_count:
        ## 排除最少用的，和最常用的
        if word_count[word] < 2000 and word_count[word] > 20:
            wordsDic.append(word)
    return wordsDic


wordsDic = create_lexicon(pos_file, neg_file)
##print(wordsDic)


def normalize_dataset(wordsDic):
    dataset = []

    # lex:词汇表；line:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(wordsDic,line, clf):
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        ##将每一条评论向量化，copy整个字典的规模，设置成1
        features = np.zeros(len(wordsDic))
        for word in words:
            if word in wordsDic:
                features[wordsDic.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(wordsDic, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(wordsDic, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            dataset.append(one_sample)

    return dataset

##pos 和 neg的向量化
dataset = normalize_dataset(wordsDic)
##　随机排序
random.shuffle(dataset)


## 挑出10%作为测试数据
test_size = int(len(dataset) * 0.1)



##　print(dataset)
##将list的,元素转成行
dataset = np.array(dataset)
# [[array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]
#  [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [1, 0]]
#  [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [1, 0]]
#  ...,
#  [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [0, 1]]
#  [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [1, 0]]
#  [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]) [1, 0]]]
#print(dataset)


## 单独看一个样本，
train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]
train_x = train_dataset[:, 0]
train_y = train_dataset[:, 1]
test_x = test_dataset[:,0]
test_y = test_dataset[:,1]

## 定义每个层有多少个神经元
n_input_layer = len(wordsDic)  # 输入层，数量与dic相同

n_layer_1 = 1000  # hide layer
n_layer_2 = 1000  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 2  # 输出层


# 两层线性
## 对每一个数据进行特征抽取，wordVector的每一个值
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


# 每次使用50条数据进行训练
batch_size = 50
## X = n行vector
X = tf.placeholder('float', [None, len(train_dataset[0][0])])
Y = tf.placeholder('float', [None, len(train_dataset[0][1])])


# print(train_x[0:50])
# print(list(train_x))

def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_func)  # learning rate 默认 0.001

    # epochs = 13
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        i = 0
        # for epoch in range(epochs):
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
            epoch_loss += c
            i += batch_size

            # print(epoch, ' : ', epoch_loss)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: list(test_x), Y: list(test_y)}))


train_neural_network(X,Y)