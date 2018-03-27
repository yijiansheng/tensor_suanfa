import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 2])
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
Y = tf.placeholder(tf.float32, [None, 1])

# 成本函数 sum(sqr(y_-y))/n
cost = tf.reduce_mean(tf.square(Y-y))

# 用梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

x_train = [[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]]
y_train = [[7], [8], [10], [14], [8], [13], [20], [16], [28], [26]]

for i in range(100):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
print("w0:%f" % sess.run(w[0]))
print("w1:%f" % sess.run(w[1]))
print("b:%f" % sess.run(b))