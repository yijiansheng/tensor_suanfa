import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#——————————————————导入数据——————————————————————
f=open('dataset_1.csv')
df=pd.read_csv(f)     #读入股票数据
##
data=np.array(df['price'])   #获取最高价序列
data=data[::-1]      #翻转
#以折线图展示data
plt.figure()
plt.plot(data)
plt.show()
## 归一化
normalize_data=(data-np.mean(data))/np.std(data)  #标准化

## 从一维到二维，从一维的一行N列，变成，N行1列
## N行一列
normalize_data=normalize_data[:,np.newaxis]


## 首先看train_x的shape
##  每一行都是20个数

##desc:
##生成训练集
#设置常量
## 二十行
time_step=20      #
##每一次训练的样例数量
batch_size=60     #每一批次训练多少个样例 60个样例？
## hidden layer
rnn_unit=10       #hidden layer units

input_size=1
output_size=1
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集
for i in range(len(normalize_data)-time_step-1):
    # 取了1-20 二十个数
    x=normalize_data[i:i+time_step]
    ## 取了2-21 二十个数
    y=normalize_data[i+1:i+time_step+1]
    ## y就是下一个x
    ## trainX有大约同样行数的元素,每一个元素是time_step的数字
    train_x.append(x.tolist())
    ## trainY每一个元素都在后一位
    train_y.append(y.tolist())




#——————————————————定义神经网络变量——————————————————
## 前面type，后面维度shape
## placeholder存样本，variables存weights和b这种变量
## 未知行数
## time_stemp表示行，后面是特征列
## X 输入未知个模型，每一个模型20行，1列
## 二十行一列组成了一个单位，train_X有N个单位 最后一个数值，代表的是特征数量
#每批次输入网络的tensor
X=tf.placeholder(tf.float32, [None,time_step,input_size])
#每批次tensor对应的标签
Y=tf.placeholder(tf.float32, [None,time_step,output_size])
#输入层、输出层权重、偏置
## 输入层权重?输出层权重?
weights={
         ##in [1,10]
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         ##out [10,1]
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        ##
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }



#——————————————————定义神经网络变量——————————————————
## 初始化神经网络
def lstm(batch):      #参数：输入多少个模型
    ## 'in':tf.Variable(tf.random_normal([input_size,rnn_unit]))
    w_in=weights['in']
    ##  'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
    b_in=biases['in']
    ##　先转成二维
    ## 先变换成只有一个特征的二维向量
    ## 以前是一次训练60个，每一个是20*1的单位
    ## 现在将20提出来
    ## X输入已经变化 timestep*n
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    ## 行保留 列变成w的列，
    input_rnn=tf.matmul(input,w_in)+b_in
    ##　
    ## reshape前， N*timestep*1个特征列
    ## reshape后， N*timestep*lstm的hidden层
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    ## lstm的隐藏层有10个
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    ## 一次输入的样例
    init_state=cell.zero_state(batch,dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，
    # final_states是最后一个cell的结果
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    ## 将X输入转换成output run_unit的特征列
    ## 至此改变了X的输入
    ## reshape到 layer个output
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    ## 最后pred是N*1
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)

    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                ##训练样本有限，一次性跨度batch_size个
                start+=batch_size
                end=start+batch_size

                #每10步保存一次参数
                if step%10==0:
                    print(i,step,loss_)
                    print("保存模型：",saver.save(sess,'stock.model'))
                step+=1


train_lstm()


#————————————————预测模型————————————————————
## 输入也要先产生变化，先经过lstm化
def prediction():
    # 预测时只输入[1,time_step,input_size]的测试数据
    pred,_=lstm(1)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('module2/')
        saver.restore(sess, module_file)

        #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        predict=[]
        #得到之后100个预测结果
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()

prediction()
