import tensorflow as tf
import numpy as np
name_dataset = 'dataset/name.csv'

train_x = []
## 用[0,1]和[1,0]作为区分
train_y = []
with open(name_dataset, 'r',encoding='utf8') as f:
    first_line = True
    for line in f:
        if first_line is True:
            first_line = False
            continue
        sample = line.strip().split(',')
        if len(sample) == 2:
            train_x.append(sample[0])
            if sample[1] == '男':
                train_y.append([0, 1])
            else:
                train_y.append([1, 0])

max_name_length = max([len(name) for name in train_x])
print("best long name lengths: ", max_name_length)
max_name_length = 8

counter = 0
vocabulary = {}
for name in train_x:
    counter += 1
    tokens = [word for word in name]
    for word in tokens:
        if word in vocabulary:
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

##　字典，最前面的出现次数最多
##　6000左右
vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
print(len(vocabulary_list))
print(vocabulary_list)
##　 给每一个字 标下标
vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
print(vocab)
## 一个name一个向量,注意构造向量的方式
train_x_vec = []
for name in train_x:
    name_vec = []
    for word in name:
        name_vec.append(vocab.get(word))
    while len(name_vec) < max_name_length:
        name_vec.append(0)
    train_x_vec.append(name_vec)

print(train_x_vec[0])

## 每个名字的向量是max_name_length
input_size = max_name_length
num_classes = 2

batch_size = 64
num_batch = len(train_x_vec) // batch_size
## n行 * 8
X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

dropout_keep_prob = tf.placeholder(tf.float32)


## 每一个字，每一个下标的描述，feature_num
embedding_size=128
## 卷积核
num_filters=128


## net
def neural_network(vocabulary_size, embedding_size=128, num_filters=128):
    # embedding layer
    ##　vocab的长度 * embedding
    W = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size],
                          -1.0, 1.0))
    ## 增加一维num_features
    embedded_chars = tf.nn.embedding_lookup(W, X)
    print(embedded_chars)
    ## 这个是INPUT输入 [batch,一个向量的长度,embedding描述,1]
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        # convolution + maxpool layer
    print(embedded_chars_expanded)
    filter_sizes = [3, 4, 5]
    ## 收集卷积结果
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        ## 卷积核的形状[ 输入的后三个, 输出的张数 ]
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
        conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
        ## 激活函数
        h = tf.nn.relu(tf.nn.bias_add(conv, b))
        print(h)
        ## 池化成 batch*1*1*128
        ##　输入的是batch*vector*embedding*1
        ##　输出的是batch*1*1*128
        pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                padding='VALID')
        print(pooled)
        pooled_outputs.append(pooled)

    ## 128 * 3
    num_filters_total = num_filters * len(filter_sizes)
    ## 扩展3列
    h_pool = tf.concat(pooled_outputs,3)
    print(h_pool)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    print(h_pool_flat)
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    with tf.name_scope("output"):
        ## 抵消W的行
        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)

    return output



## training process
def train_neural_network():
    output = neural_network(len(vocabulary_list))
    optimizer = tf.train.AdamOptimizer(0.01)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())

    print(num_batch)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_batch):
            batch_x = train_x_vec[i * batch_size: (i + 1) * batch_size]
            batch_y = train_y[i * batch_size: (i + 1) * batch_size]
            _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
            print(i,"-----",loss_)
            # 保存模型
        if i == (num_batch-1):
            saver.save(sess, "tmp/name2sex.model")
            print("training finish")

### train_neural_network()


## model predict
def detect_sex(name_list):
    x = []
    for name in name_list:
        name_vec = []
        for word in name:
            name_vec.append(vocab.get(word))
        while len(name_vec) < max_name_length:
            name_vec.append(0)
        ##　ｘ是输入的batch的向量
        x.append(name_vec)


    output = neural_network(len(vocabulary_list))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        saver.restore(sess, 'tmp/name2sex.model')
        predictions = tf.argmax(output, 1)
        res = sess.run(predictions, {X: x, dropout_keep_prob: 1.0})
        i = 0
        for name in name_list:
            print(name, '女' if res[i] == 0 else '男')
            i += 1


detect_sex(["李超", "李思思"])