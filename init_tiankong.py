import re
import random
import ast
import itertools
import pickle
import numpy as np


train_data_file = 'dataset/cbtest_NE_train.txt'
valid_data_file = 'dataset/cbtest_NE_valid_2000ex.txt'


def preprocess_data(data_file, out_file):
    stories = []
    with open(data_file) as f:
        story = []
        j = 0
        for i in range(50000):
            line = f.readline()
            line = line.strip()
            if not line:
                story = []
            else:
                _, line = line.split(' ', 1)
                if line:
                    if '\t' in line:
                        j = j+1
                        print(j)
                        q, a, _, answers = line.split('\t')
                        q = [s.strip() for s in re.split('(\W+)+', q) if s.strip()]
                        stories.append((story, q, a))
                    else:
                        line = [s.strip() for s in re.split('(\W+)+', line) if s.strip()]
                        story.append(line)

    ## 样本
    samples = []
    for story in stories:
        story_tmp = []
        content = []
        for c in story[0]:
            content += c
        story_tmp.append(content)
        story_tmp.append(story[1])
        story_tmp.append(story[2])
        samples.append(story_tmp)

    random.shuffle(samples)

    with open(out_file, "w") as f:
        for sample in samples:
            f.write(str(sample))
            f.write('\n')

## preprocess_data(train_data_file, 'dataset/tk_train.data')
## preprocess_data(valid_data_file, 'dataset/tk_valid.data')



def read_data(data_file):
    stories = []
    with open(data_file) as f:
        for line in f:
            line = ast.literal_eval(line.strip())
            stories.append(line)
    return stories


stories = read_data('dataset/tk_train.data') + read_data('dataset/tk_valid.data')

content_length = max([len(s) for s, _, _ in stories])
question_length = max([len(q) for _, q, _ in stories])
## 985的content_length   156的question_length
print(content_length, question_length)


vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))
vocab_size = len(vocab) + 1
print(vocab_size)
word2idx = dict((w, i + 1) for i,w in enumerate(vocab))
## 排序之后的词汇表
print(word2idx)
pickle.dump((word2idx, content_length, question_length, vocab_size), open('dataset/tk_vocab.data', "wb"))


## 对于不够的story ，进行补充
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)


        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x



def to_vector(data_file, output_file):
    word2idx, content_length, question_length, _ = pickle.load(open('dataset/tk_vocab.data', "rb"))
    X = []
    Q = []
    A = []
    with open(data_file) as f_i:
        for line in f_i:
            line = ast.literal_eval(line.strip())
            x = [word2idx[w] for w in line[0]]
            q = [word2idx[w] for w in line[1]]
            a = [word2idx[line[2]]]

            X.append(x)
            Q.append(q)
            A.append(a)

    X = pad_sequences(X, content_length)
    Q = pad_sequences(Q, question_length)
    with open(output_file, "w") as f_o:
        for i in range(len(X)):
            f_o.write(str([X[i].tolist(), Q[i].tolist(), A[i]]))
            f_o.write('\n')

## 最后形成的结果,2000+行，content里面每个字用序号的向量表示,question也是,answer也是
to_vector('dataset/tk_train.data', 'dataset/tk_train.vec')
to_vector('dataset/tk_valid.data', 'dataset/tk_valid.vec')


