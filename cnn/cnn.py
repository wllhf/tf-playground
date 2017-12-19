import tensorflow as tf


from util.elements import conv2d_layer, max_pool_layer, fc_layer


class cnn(object):
    """
    Convolutional neural network.

    1. Convolutional layer + max pooling,
    2. Convolutional layer + max pooling,
    3. Fully connected layer \w dropout
    4. Fully connected output layer

    MNIST: > 95 %
    """

    def __init__(self, x, y, filter_size=[7, 7], dim_1conv=32, dim_2conv=64, dim_ful=1024):

        psize = [x.get_shape()[1].value, x.get_shape()[2].value]
        nchannels = x.get_shape()[3].value
        nclasses = y.get_shape()[1].value
        dim_1ful_in = psize[0] // 4 * psize[1] // 4 * dim_2conv

        # placeholders
        self._input = x
        self._target = y

        # inference
        with tf.name_scope('core_network'):
            with tf.variable_scope('1st_layer'):
                h_conv1 = conv2d_layer(self._input, filter_size, nchannels, dim_1conv)
                h_pool1 = max_pool_layer(h_conv1, ps=2, ss=2)

            with tf.variable_scope('2nd_layer'):
                h_conv2 = conv2d_layer(h_pool1, filter_size, dim_1conv, dim_2conv)
                h_pool2 = max_pool_layer(h_conv2, ps=2, ss=2)

            with tf.variable_scope('fc_layer'):
                self._keep_prob = tf.placeholder(tf.float32)
                h_pool2_flat = tf.reshape(h_pool2, [-1, dim_1ful_in])
                h_fc1 = fc_layer(h_pool2_flat, dim_1ful_in, dim_ful, dropout=self._keep_prob)

            with tf.variable_scope('output_layer'):
                self._prediction = fc_layer(h_fc1, dim_ful, nclasses, act=tf.identity)

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
