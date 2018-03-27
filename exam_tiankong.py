import tensorflow as tf
import pickle
import numpy as np
import ast
from collections import defaultdict




## 待梳理
train_data = 'dataset/tk_train.vec'
valid_data = 'dataset/tk_valid.vec'
word2idx, content_length, question_length, vocab_size = pickle.load(open('dataset/tk_vocab.data', "rb"))
print(content_length, question_length, vocab_size)

batch_size = 64
train_file = open(train_data)

def get_next_batch():
    X = []
    Q = []
    A = []
    for i in range(batch_size):
        for line in train_file:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
            break

    if len(X) == batch_size:
        return X, Q, A
    else:
        train_file.seek(0)
        return get_next_batch()

## [0]是batch个content_length [1]是batch个question_length
def get_test_batch():
    with open(valid_data) as f:
        X = []
        Q = []
        A = []
        for line in f:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
        return X, Q, A

## batch_size * content 不足content的用0
X = tf.placeholder(tf.int32, [batch_size, content_length])
Q = tf.placeholder(tf.int32, [batch_size, question_length])
A = tf.placeholder(tf.int32, [batch_size])

keep_prob = tf.placeholder(tf.float32)

def glimpse(weights, bias, encodings, inputs):
    weights = tf.nn.dropout(weights, keep_prob)
    inputs = tf.nn.dropout(inputs, keep_prob)
    attention = tf.transpose(tf.matmul(weights, tf.transpose(inputs)) + bias)
    attention = tf.matmul(encodings, tf.expand_dims(attention, -1))
    attention = tf.nn.softmax(tf.squeeze(attention, -1))
    return attention, tf.reduce_sum(tf.expand_dims(attention, -1) * encodings, 1)



def neural_attention(embedding_dim=384, encoding_dim=128):
    ## embedding
    embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_dim], stddev=0.22), dtype=tf.float32)
    # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), [embeddings])

    with tf.variable_scope('encode'):
        with tf.variable_scope('X'):
            ## reduc_sum( ,1) 一行一个
            X_lens = tf.reduce_sum(tf.sign(tf.abs(X)), 1)
            print("X")
            print(X)
            print(X_lens)
            ## 给X添加向量，X的一个值是一个index，给这个index添加描述
            embedded_X = tf.nn.embedding_lookup(embeddings, X)
            encoded_X = tf.nn.dropout(embedded_X, keep_prob)
            print(encoded_X)
            gru_cell = tf.contrib.rnn.GRUCell(encoding_dim)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, encoded_X,
                                                                     sequence_length=X_lens, dtype=tf.float32,
                                                                     swap_memory=True)
            ## 两部分，两个tensor
            print(outputs)
            print(outputs[-1])
            encoded_X = tf.concat(outputs,2)
            print(encoded_X)
        with tf.variable_scope('Q'):
            Q_lens = tf.reduce_sum(tf.sign(tf.abs(Q)), 1)
            embedded_Q = tf.nn.embedding_lookup(embeddings, Q)
            encoded_Q = tf.nn.dropout(embedded_Q, keep_prob)
            gru_cell = tf.contrib.rnn.GRUCell(encoding_dim)
            ## 双向,变最后一维
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell,encoded_Q,
                                                                     sequence_length=Q_lens, dtype=tf.float32,
                                                                     swap_memory=True)
            print("Q")
            print(outputs)
            ## 合并
            encoded_Q = tf.concat(outputs,2)
            print(encoded_Q)

    #
    W_q = tf.Variable(tf.random_normal([2 * encoding_dim, 4 * encoding_dim], stddev=0.22), dtype=tf.float32)
    b_q = tf.Variable(tf.random_normal([2 * encoding_dim, 1], stddev=0.22), dtype=tf.float32)
    W_d = tf.Variable(tf.random_normal([2 * encoding_dim, 6 * encoding_dim], stddev=0.22), dtype=tf.float32)
    b_d = tf.Variable(tf.random_normal([2 * encoding_dim, 1], stddev=0.22), dtype=tf.float32)
    g_q = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)
    g_d = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)

    with tf.variable_scope('attend') as scope:
        ## 推算
        infer_gru = tf.contrib.rnn.GRUCell(4 * encoding_dim)
        infer_state = infer_gru.zero_state(batch_size, tf.float32)
        for iter_step in range(8):
            if iter_step > 0:
                scope.reuse_variables()

            _, q_glimpse = glimpse(W_q, b_q, encoded_Q, infer_state)
            d_attention, d_glimpse = glimpse(W_d, b_d, encoded_X, tf.concat([infer_state, q_glimpse], 1))

            gate_concat = tf.concat([infer_state, q_glimpse, d_glimpse, q_glimpse * d_glimpse], 1)

            r_d = tf.sigmoid(tf.matmul(gate_concat, g_d))
            r_d = tf.nn.dropout(r_d, keep_prob)
            r_q = tf.sigmoid(tf.matmul(gate_concat, g_q))
            r_q = tf.nn.dropout(r_q, keep_prob)

            combined_gated_glimpse = tf.concat([r_q * q_glimpse, r_d * d_glimpse], 1)
            _, infer_state = infer_gru(combined_gated_glimpse, infer_state)

    return tf.to_float(tf.sign(tf.abs(X))) * d_attention

neural_attention()
# print(len(get_next_batch()[0][1]))
# print(len(get_next_batch()[1][1]))