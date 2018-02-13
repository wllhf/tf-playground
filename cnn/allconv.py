""" https://arxiv.org/abs/1412.6806 """

import tensorflow as tf

from ..util.elements import conv2d_layer


class allconv(object):
    """
    2014: All convolutional network. Training process is simplified using AdamOptimizer.

    MNIST ~97%
    CIFAR10 ~83% (200 epochs, original paper ~91% at 350 epochs)
    """

    def __init__(self, x, y):
        nch = x.get_shape()[3].value
        ncl = y.get_shape()[1].value

        # placeholders
        self._input = x
        self._target = y
        self._keep_prob = tf.placeholder(tf.float32)
        self._keep_prob_in = tf.placeholder(tf.float32)

        # network
        with tf.name_scope('core_network'):
            with tf.variable_scope('dropout_in'):
                x = tf.nn.dropout(x, self._keep_prob_in)
            with tf.variable_scope('33conv96_1'):
                x = conv2d_layer(x, [3, 3], nch,  96)
            with tf.variable_scope('33conv96_2'):
                x = conv2d_layer(x, [3, 3],  96,  96)
            with tf.variable_scope('33conv96_subsampling'):
                x = conv2d_layer(x, [3, 3],  96,  96, strides=[1, 2, 2, 1], dropout=self._keep_prob)
            with tf.variable_scope('33conv192_1'):
                x = conv2d_layer(x, [3, 3],  96, 192)
            with tf.variable_scope('33conv192_2'):
                x = conv2d_layer(x, [3, 3], 192, 192)
            with tf.variable_scope('33conv192_subsampling'):
                x = conv2d_layer(x, [3, 3], 192, 192, strides=[1, 2, 2, 1], dropout=self._keep_prob)
            with tf.variable_scope('33conv192_3'):
                x = conv2d_layer(x, [3, 3], 192, 192)
            with tf.variable_scope('11conv192_1'):
                x = conv2d_layer(x, [1, 1], 192, 192)
            with tf.variable_scope('11conv10_1'):
                x = conv2d_layer(x, [1, 1], 192, ncl)
            with tf.variable_scope('averaging'):
                x = tf.layers.average_pooling2d(x, x.get_shape()[1:3], [1, 1])
                self._prediction = tf.reshape(x, [-1, ncl])

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

        # train
        with tf.name_scope('train_op'):
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
        return (self._input, self._target, self._keep_prob, self._keep_prob_in)


def train_and_test(trn, tst, mbatch_size=50, epochs=20, log_dir='./log/'):
    from math import ceil
    from ..util.util import mkrundir

    tf.reset_default_graph()
    run_dir = mkrundir(log_dir)

    samples, labels = trn
    patch_size = samples.shape[1:]
    nchannels = patch_size[2] if len(patch_size) > 2 else 1
    nclasses = labels.shape[1]
    nsamples = samples.shape[0]
    iterations = ceil(nsamples / mbatch_size)

    # graph
    x = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], nchannels], name='input_layer')
    y = tf.placeholder(tf.float32, shape=[None, nclasses], name='target')

    model = allconv(x, y)
    kp_in, kp = model.io_placeholder[2:]

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    # training
    for epoch in range(epochs):

        for i in range(ceil(trn[0].shape[0] / mbatch_size)):
            s, e = i * mbatch_size, (i + 1) * mbatch_size
            feed_dict = {x: trn[0][s:e, :].astype('float32'), y: trn[1][s:e, :], kp_in: 0.2, kp: 0.5}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            feed_dict = {x: tst[0][:mbatch_size, :].astype('float32'), y: tst[1][:mbatch_size, :], kp_in: 1.0, kp: 1.0}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    nsamples = samples.shape[0]
    iterations = ceil(nsamples / mbatch_size)

    results = []
    for i in range(iterations):
        s, e = i * mbatch_size, (i + 1) * mbatch_size
        feed_dict = {x: tst[0][s:e, :].astype('float32'), y: tst[1][s:e, :], kp_in: 1.0, kp: 1.0}
        results.append(model.evaluation.eval(feed_dict=feed_dict))

    sess.close()
    return sum(results)/len(results)


if __name__ == '__main__':
    from ..util.data import load_mnist
    from ..util.data import load_cifar10

    log_dir = "./log"

    print("Train and test on MNIST:")
    DATA_DIR = "~/data/mnist"
    trn, tst = load_mnist(DATA_DIR)
    train_and_test(trn, tst, epochs=20, log_dir=log_dir)

    print("Train and test on CIFAR10:")
    DATA_DIR = "~/data/cifar10_py"
    trn, tst = load_cifar10(DATA_DIR, flatten=False)
    train_and_test(trn, tst, epochs=200, log_dir=log_dir)
