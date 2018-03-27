import collections
import numpy as np
import tensorflow as tf
poetry_file = 'dataset/poetry.txt'

# 每个样本存在关系
poetrys = []
with open(poetry_file, "r", encoding='utf-8', ) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

## poetrys是一个list
poetrys = sorted(poetrys, key=lambda line: len(line))

all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
## 一共由多少item组成的
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
## 按照出现的次数，排序出item
words, _ = zip(*count_pairs)
## 加上一个space
words = words[:len(words)] + (' ',)
## key是word ,value是序号
word_num_map = dict(zip(words, range(len(words))))
to_num = lambda word: word_num_map.get(word, len(words))
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
## 将这个poetry全部变成向量
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]

## 每次取10个向量
batch_size =1

n_block = len(poetrys_vector)//batch_size
x_batch=[]
y_batch=[]



for i in range(n_block):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batches = poetrys_vector[start_index:end_index]
    ##这一组最大的行
    length = max(map(len,batches))
    ## 填满
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:,:-1] = xdata[:,1:]
    x_batch.append(xdata)
    y_batch.append(ydata)


print(n_block)
print(len(words))
## words 是已经排好序的
print(words)
## x_batch n_block * batchsize行 * 每一组最大的列数
# print(x_batch[1000].shape)
# print(y_batch[1000].shape)

## 10行 * 未知列
input_data = tf.placeholder(tf.int32, [batch_size, None])
## 10行，未知列
output_targets = tf.placeholder(tf.int32, [batch_size, None])


## 特征值128个，两层cell
def neural_network(feature_num=128, num_layer=2):
    cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(feature_num, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layer, state_is_tuple=True)
    ## 属于优化,定义初始状态
    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("rnnlm"):
        ##     全连接 接上 输出 w: [feature , 列len(words) ]
        ##     最后求解就是词组向量的权重
        softmax_w = tf.get_variable("softmax_w", [feature_num, len(words) + 1])
        print(softmax_w)
        ##
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        print(softmax_b)
        ## 所有字这么多行 ,每一个字128个特征
        ## 词组向量
        embedding = tf.get_variable("embedding", [len(words) + 1, feature_num])
        print(embedding)

        ## 输入的input_data [10 , num_steps ]未知列，其中每一个元素是一个index 将这个index变成词向量
        ## inputs = [batchsize * ? * vector的feature]
        ## 这里注意，一条长序列进来，成为一定shape的矩阵，每一个点表示一个词的下标，近似认识为那个字
        ## 但是这个字，不是向量，所以要抽象成为向量描述
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        print(input_data)
        print(inputs)
    ## 开始时刻的输入 和 初始化状态
    ## 代码的意思是： inputs[所有batch ，某一列 ,  词向量  ]
    ## 什么样shape的输入，什么样的输出
    ## output [batchsize , numstep , hidden_feature个特征]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    print(outputs)
    ## reshape成为[batch_size*steps,feature_num]
    ## 逆向折叠，相当于一个长序列的每一个词，后面跟了feature_num个特征描述
    output = tf.reshape(outputs, [-1, feature_num])
    print(output)
    ## [全部进来的字 * words.length的向量]
    logits = tf.matmul(output, softmax_w) + softmax_b
    print(logits)
    probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state



def train_nerual_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1])
    #    loss = tf.contrib.seq2seq.seq2seq_loss([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    ## logits是一个 扩展 words.length的向量 6110个描述
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                              ## 期待的正确答案 [batch * steps]压缩了一维
                                                              [targets],
                                      [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = 0.1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    ##进入session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        n = 0
        while n<(n_block-30):
            train_loss, _, _ = sess.run([cost, last_state, train_op],
                                        feed_dict={input_data: x_batch[n], output_targets: y_batch[n]})
            print(n, train_loss)
            n += 20
        saver.save(sess, "D://do//new3//tensor_suanfa//tmp//poetry.module")



##train_nerual_network()


def gen_poetry():
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return words[sample]

    _, last_state, probs, cell, initial_state = neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, "D://do//new3//tensor_suanfa//tmp//./poetry.module")
        state_ = sess.run(cell.zero_state(1, tf.float32))

        x = np.array([list(map(word_num_map.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        poem = ''
        while word != ']':
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_num_map[word]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
    return poem

#train_nerual_network()
print(gen_poetry())