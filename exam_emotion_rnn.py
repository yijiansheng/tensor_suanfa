from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import tensorflow as tf

##　注意有分割输入的概念
## 输入层
chunk_size = 28
chunk_n = 28

rnn_size = 256
n_output_layer = 10  # 输出层
X = tf.placeholder('float', [None, chunk_n, chunk_size])
Y = tf.placeholder('float', [None, n_output_layer])


def recurrent_neural_network(data):
    ## 全连接层 run_size
    layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             'b_': tf.Variable(tf.random_normal([n_output_layer]))}
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(0, chunk_n, data)
    outputs, status = tf.nn.rnn(lstm_cell, data, dtype=tf.float32)

    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])

    return ouput

recurrent_neural_network(X)