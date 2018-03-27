import numpy as np
import tensorflow as tf

x= np.array([[1,2],[3,4]])
y= np.array([[1,2],[3,4]])
result = tf.multiply(x,y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(result)
    print(res)