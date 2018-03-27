import tensorflow as tf
import numpy as np
import pandas as pd



features_num = 4
col_y_num = 1
y_index=3
## 每次输入进10个
time_step = 10
f=open('dataset/dongguan_features.csv')
df=pd.read_csv(f,names=['per_pv','per_uv','per_order','per_zhuanhua'])
data=np.array(df)
## 这样的data是N行M列的矩阵
train_data=data[1:,:]
## 保持矩阵行列性，注意行列之间用:来控制
x_features = train_data[0:len(train_data)-1,:]
y_features = train_data[1:len(train_data),col_y_num:col_y_num+1]
x_features = x_features.astype(np.float64)
y_features = y_features.astype(np.float64)


##归一化
##每一个特征都做归一化
max_hang = x_features.max(axis=0)
min_hang = x_features.min(axis=0)
diff_hang = max_hang-min_hang
x_features = x_features/diff_hang

##选择time_step行，预测time_step+1行
all_hang = len(x_features)
all_lie = len(x_features[0])
x_train = np.zeros([all_hang-time_step+1,features_num])
y_train = np.zeros([all_hang-time_step+1,col_y_num])

##time_stemp的平均值，预测time_stemp+1行的y值
for index in range(all_hang-time_step+1):
    temp=x_features[index:index+time_step,:]
    x_train[index,:] = np.mean(temp, axis=0)
    y_train[index,:] = y_features[index+time_step-1]

# for i in range(pv_data.size-2):
#     x_train.append(pv_data[i].tolist())
#     y_train.append(pv_data[i+1].tolist())

# 取最后面十行值
x_test = train_data[len(train_data)-time_step:len(train_data),:]
x_test = x_test.astype(np.float64)
x_test = x_test/diff_hang
x_test = np.mean(x_test,axis=0)
x_test = x_test[np.newaxis,:]


X = tf.placeholder(tf.float32, [None, features_num])
w = tf.Variable(tf.zeros([features_num, col_y_num]))
b = tf.Variable(tf.zeros([col_y_num]))
y = tf.matmul(X, w) + b
Y = tf.placeholder(tf.float32, [None, col_y_num])

cost = tf.reduce_mean(tf.sqrt(tf.sqrt(tf.square(Y-y))))
#cost = tf.reduce_mean(tf.sqrt(tf.sqrt(tf.square(Y-y))))
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