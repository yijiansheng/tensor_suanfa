import tensorflow as tf
import numpy as np
import cv2

## 这是已经训练好的pattern
inception_model = 'dataset/tensorflow_inception_graph.pb'
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

##　随机挑选一个张量
X = tf.placeholder(np.float32, name='input')
with tf.gfile.FastGFile(inception_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print("read finish")
## 数据集均值
imagenet_mean = 117.0
## 增加一维 0 指的是在最前面增加一维
preprocessed = tf.expand_dims(X-imagenet_mean, 0)


## input_map
tf.import_graph_def(graph_def, {'input':preprocessed})
## 获取网络的层数 59层 ，随机选出一层，看pattern
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
## 总通道数
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

##　这个神经网络一共有５９层
print('layers:', len(layers))   # 59
print('feature:', sum(feature_nums))  # 7548


def deep_dream(obj,
   ##　这个img_noise的意思，是先生成一个噪声图像
    img_noise=np.random.uniform(size=(224, 224, 3)) + 100.0, iter_n=10,
               step=1.5, octave_n=4,octave_scale=1.4):
    ##　定义最优objective
    score = tf.reduce_mean(obj)
    gradi = tf.gradients(score, X)[0]
    ## 噪声图像
    img = img_noise
    octaves = []

    ## 多尺度图像的生成
    ## 将TF图生成函数 转换为规则的
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap


    ##　使用tf调整大小的辅助函数
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)
    for _ in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = sess.run(t_grad, {X: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, gradi)
            img += g * (step / (np.abs(g).mean() + 1e-7))

            # 保存图像
        output_file = 'tmp/output' + str(octave + 1) + '.jpg'
        cv2.imwrite(output_file, img)
        print(output_file)


# 加载输入图像
input_img = cv2.imread('tmp/input1.jpg')
input_img = np.float32(input_img)

# 选择层
layer = 'mixed4c'

deep_dream(tf.square(graph.get_tensor_by_name("import/%s:0" % layer)), input_img)