from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import tensorflow as tf
import numpy as np
import scipy.misc as misc

## 从噪音推img
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    print(G_prob)
    return G_prob

## 鉴别
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
D_W1 = tf.Variable(tf.truncated_normal([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
D_W2 = tf.Variable(tf.truncated_normal([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')
theta_D = [D_W1, D_W2, D_b1, D_b2]


## 噪音，行数一定，100特征
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
# Generator Net
G_W1 = tf.Variable(tf.truncated_normal([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
G_W2 = tf.Variable(tf.truncated_normal([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]


G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)
batch_size = 50

real_loss = tf.sqrt(2 * tf.nn.l2_loss(D_real - X)) / batch_size
fake_loss = tf.sqrt(2 * tf.nn.l2_loss(D_fake - G_sample)) / batch_size
margin = 20
D_loss = margin - fake_loss + real_loss
G_loss = fake_loss

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))



n_dim = 100
D_optimizer = tf.train.AdamOptimizer(0.0001).minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list=theta_G)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(5000):
        train_x, _ = mnist.train.next_batch(batch_size)
        batch_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, n_dim]).astype(np.float32)
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={X: train_x, Z: batch_noise})
        _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={Z: batch_noise})
        print(it,D_loss_curr,G_loss_curr)

        if it == 1000:
            ## 5张
            test_noise = np.random.uniform(-1.0, 1.0, size=(5, n_dim)).astype(np.float32)
            images = sess.run(G_sample, feed_dict={Z: test_noise})
            for k in range(5):
                image = images[k,:]
                # image += 1
                # image *= 127.5
                # image = np.clip(image, 0, 255).astype(np.uint8)
                image = np.reshape(image, (28, 28))
                misc.imsave('tmp/img/word_image' + str(k) + '.jpg', image)
