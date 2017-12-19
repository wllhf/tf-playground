import tensorflow as tf

from util.elements import conv2d_layer, max_pool_layer, fc_layer


class lenet(object):
    """
    Simplified version of LeNet-5. Cross entropy on logits instead
    of RBFs. All feature maps between S2 and C3 connected. First
    convolution is padded to use network on unadjusted MNIST data.
    """

    def __init__(self, x, y):

        nchannels = x.get_shape()[3].value
        nclasses = y.get_shape()[1].value

        # placeholders
        self._input = x
        self._target = y

        # inference
        with tf.name_scope('core_network'):
            with tf.variable_scope('C1'):
                c1 = conv2d_layer(self._input, [5, 5], nchannels, 6, act=tf.identity)
            with tf.variable_scope('S2'):
                s2 = max_pool_layer(c1, ps=2, ss=2)
            with tf.variable_scope('C3'):
                c3 = conv2d_layer(s2, [5, 5], 6, 16, padding='VALID', act=tf.identity)
            with tf.variable_scope('S4'):
                s4 = max_pool_layer(c3, ps=2, ss=2)
            with tf.variable_scope('C5'):
                c5 = conv2d_layer(s4, [5, 5], 16, 120, padding='VALID', act=tf.identity)
                c5 = tf.reshape(c5, [-1, 120])
            with tf.variable_scope('F6'):
                f6 = fc_layer(c5, 120, 84, act=tf.tanh)
            with tf.variable_scope('OUTPUT'):
                self._prediction = fc_layer(f6, 84, nclasses, act=tf.identity)

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
        return (self._input, self._target)

