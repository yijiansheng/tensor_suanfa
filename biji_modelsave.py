from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "D://do//new3//tensor_suanfa//tmp//modelsave.ckpt"

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

## model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    ## 线性之后跟一个非线性因素
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
## arg这个函数返回的是最大值下标
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.global_variables_initializer()
## 一个保存器
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                      y: batch_y})


    print("First Optimization Finished!")
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    ##保存的是sess
    save_path = saver.save(sess, "D://do//new3//tensor_suanfa//tmp//modelsave.ckpt")
    print ("Model saved in file: %s" % save_path)

## 再开一个session
print("Starting 2nd session...")
with tf.Session() as sess:
    sess.run(init)
    load_path = saver.restore(sess, "D://do//new3//tensor_suanfa//tmp//./modelsave.ckpt")
    print("Model restored from file: %s" % save_path)
    for i in range(200):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                      y: batch_y})
    print("Second Optimization Finished!")
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))