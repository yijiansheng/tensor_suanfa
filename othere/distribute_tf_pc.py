# coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("data_dir", "", "data dir")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size of training data")
tf.app.flags.DEFINE_string("log_dir", "", "log dir")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("gpu_index", 0, "Index of gpu within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name="worker", task_index=FLAGS.task_index, config=config)

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        input = tf.placeholder("float")
        label = tf.placeholder("float")

        weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
        biase = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
        pred = tf.add(tf.mul(input, weight), biase)

        loss_value = loss(label, pred)

        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value, global_step=global_step)
        init_op = tf.initialize_all_variables()

        saver = tf.train.Saver()
        tf.scalar_summary('cost', loss_value)
        summary_op = tf.merge_all_summaries()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir=FLAGS.log_dir,
                             init_op=init_op,
                             summary_op=None,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=60)
    with sv.managed_session(server.target) as sess:
        step = 0
        x, y = load(FLAGS.data_dir)
        for epoch in range(steps_to_validate):
            for (train_x, train_y) in zip(x, y):
                sess.run(train_op, feed_dict={input: train_x, label: train_y})
                if epoch % 2 == 0:
                    print (str(sess.run(weight)) + ' ' + str(sess.run(biase)))
    sv.stop()


def loss(label, pred):
    return tf.square(label - pred)


def load(data_dir):
    train_x, train_y = [], []
    f = open(data_dir, 'r')
    for line in f.readlines():
        ss = line.strip().split(' ')
        if len(ss) == 2:
            train_x.append(float(ss[0]))
            train_y.append(float(ss[1]))
    f.close()
    return train_x, train_y


if __name__ == "__main__":
    tf.app.run()