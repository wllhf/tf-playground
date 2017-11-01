import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight_variable')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias_variable')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# def fully_connected_layer(input_tensor, dim_in, dim_out, layer_name, act=tf.nn.relu):
#     with tf.name_scope(layer_name):
#         W = weight_variable([dim_in, dim_out])
#         b = bias_variable([dim_out])
#         return act(tf.matmul(input_tensor, W) + b)


class cnn(object):
    """ """

    def __init__(self):

        PATCH_SIZE = [28, 28]
        NCHANNELS = 1
        NCLASSES = 10

        FILTER_SIZE = [7, 7]
        DIM_1L = 32
        DIM_2L = 64
        DIM_FUL = 1024

        # placeholders
        with tf.name_scope('input'):
            self._input = tf.placeholder(tf.float32,
                                         shape=[None, PATCH_SIZE[0], PATCH_SIZE[1], NCHANNELS],
                                         name='input_layer')
        with tf.name_scope('target'):
            self._target = tf.placeholder(tf.float32,
                                          shape=[None, NCLASSES],
                                          name='target')

        print(self._input)
        # inference
        with tf.name_scope('core_network'):
            with tf.name_scope('1st_layer'):
                W_conv1 = weight_variable([FILTER_SIZE[0], FILTER_SIZE[1], NCHANNELS, DIM_1L])
                b_conv1 = bias_variable([DIM_1L])
                h_conv1 = tf.nn.relu(conv2d(self._input, W_conv1) + b_conv1)
                h_pool1 = max_pool_2x2(h_conv1)

            with tf.name_scope('2nd_layer'):
                W_conv2 = weight_variable([FILTER_SIZE[0], FILTER_SIZE[1], DIM_1L, DIM_2L])
                b_conv2 = bias_variable([DIM_2L])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)

            with tf.name_scope('fc_layer'):
                W_fc1 = weight_variable([PATCH_SIZE[0] // 4 * PATCH_SIZE[1] // 4 * DIM_2L, DIM_FUL])
                b_fc1 = bias_variable([DIM_FUL])
                h_pool2_flat = tf.reshape(h_pool2, [-1, PATCH_SIZE[0] // 4 * PATCH_SIZE[1] // 4 * DIM_2L])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            with tf.name_scope('dropout'):
                self._keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

            with tf.name_scope('output'):
                W_fc2 = weight_variable([DIM_FUL, NCLASSES])
                b_fc2 = bias_variable([NCLASSES])
                self._prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

        # train
        with tf.name_scope('train_op'):
            # self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)
            self._train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # evaluation
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._prediction, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self._accuracy)

        self._merged = tf.summary.merge_all()

    @property
    def prediction(self):
        return self._prediction

    @property
    def loss(self):
        return self._loss

    @property
    def train(self):
        return self._train

    @property
    def evaluation(self):
        return self._accuracy

    @property
    def summary(self):
        return self._merged

    @property
    def io_placeholder(self):
        return (self._input, self._target, self._keep_prob)
