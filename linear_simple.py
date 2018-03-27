import tensorflow as tf
import numpy as np
import pandas as pd

f=open('dataset/guangzhou_dongguan.csv')
df=pd.read_csv(f)
data=np.array(df['per_uv'])
pv_data=data[:,np.newaxis]
x_train = pv_data[0:pv_data.size-1,:]
y_train = pv_data[1:pv_data.size,:]
# for i in range(pv_data.size-2):
#     x_train.append(pv_data[i].tolist())
#     y_train.append(pv_data[i+1].tolist())
## 取最近一个值
x_test = pv_data[pv_data.size-2:pv_data.size-1,:]

X = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
Y = tf.placeholder(tf.float32, [None, 1])


# 成本函数 sum(sqr(y_-y))/n
## 尽量不要挑选太严格的cost方式
#cost = tf.reduce_mean(tf.sqrt(tf.sqrt(tf.square(Y-y))))
cost = tf.reduce_mean(tf.sqrt(tf.sqrt(tf.square(Y-y))))
# 用梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

y_test = tf.matmul(X, w) + b

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(10000):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
print(sess.run(w))
print(sess.run(b))
print(sess.run(y_test,feed_dict={X:x_test,Y:y_train}))