import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
training_iters = 2000
display_step = 10

n_input = 784
n_classes = 10
dropout = 0.75

## 未知行数
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
## 待处理
keep_prob = tf.placeholder(tf.float32)


## conv & relu
def conv2d(x, W, b, strides=1):
    ## same是表示越过边界，W是一个卷积窗口函数，要做叉积处理
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

## pool
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

weights = {
    # 5x5 conv
    ## 刚进来的时候minst的imageRgb是只有1的
    ## 32代表着32个 5*5
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv
    ## 32张进来，64张出去
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    ## 最后平铺是7*7*64,设置1024个神经元,保留1024
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    ## 1024个输入，最后输出的是10的一维向量
    ## 1024行，10列
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}




## 最后留列，给每一个特征加上一个偏移
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



## create_model
# def conv_net(x, weights, biases, dropout):
#     ## 变换形状 将50*784
#     x = tf.reshape(x, shape=[-1, 28, 28, 1])
#     ## 留32
#     conv1 = conv2d(x, weights['wc1'], biases['bc1'])
#     ## 池化
#     conv1 = maxpool2d(conv1, k=2)
#
#     conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
#     conv2 = maxpool2d(conv2, k=2)
#
#     ## 能够全连接的层
#     fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#     fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#     ## 全连接之后激活
#     fc1 = tf.nn.relu(fc1)
#     ## 在训练的时候开启，废弃掉一些神经元
#     fc1 = tf.nn.dropout(fc1, dropout)
#     ## 输出 10列
#     out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
#     print(out.shape)
#     return out






## 变换形状 将50*784
x_shape1 = tf.reshape(x, shape=[-1, 28, 28, 1])
## 留32
## 从这里开始，50行一直没有变过，一直是列在变换
## 可见conv不会影响开始的输入个数
conv1 = conv2d(x_shape1, weights['wc1'], biases['bc1'])
## 池化
conv1 = maxpool2d(conv1, k=2)

conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
conv2 = maxpool2d(conv2, k=2)

## 能够全连接的层
fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
## 全连接之后激活
fc1 = tf.nn.relu(fc1)
## 在训练的时候开启，废弃掉一些神经元
fc1 = tf.nn.dropout(fc1, dropout)
## 输出 10列
out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])


## 描述graph
pred = out

print(y.shape)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

## 训练
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()



## run graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        batch = mnist.train.next_batch(50)
        if i == 10:
            train_accuracy = sess.run(accuracy,feed_dict={
                                                    x: batch[0],
                                                    y: batch[1],
                                                    keep_prob: 1.0})
            train_pred = sess.run(pred,feed_dict={
                x: batch[0],
                y: batch[1],
                keep_prob: 1.0
            })

            train_fc1 = sess.run(fc1,feed_dict={
                x: batch[0],
                y: batch[1],
                keep_prob: 1.0
            })

            train_conv1 = sess.run(conv1,feed_dict={
                x: batch[0],
                y: batch[1],
                keep_prob: 1.0
            })
            # 打印
            print ("step %d, training accuracy %g" % (i, train_accuracy))
            print(train_conv1.shape)
            print(train_fc1.shape)
            print(train_pred.shape)
        # 执行训练模型
        sess.run(optimizer,feed_dict={x: batch[0],
                                      y: batch[1],
                                      keep_prob: dropout})