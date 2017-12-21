""" """

import tensorflow as tf

from util.elements import conv2d_layer, max_pool_layer, fc_layer


class vggnet(object):
    """ VGG network (16 layer version).
    Year         : 2014
    Parameters   : ~ 138 mio
    Contribution : Depth is important.

    Notes:
    Like all modules in this repository, this is a modified version of the original.
    """

    def __init__(self, x, y):
        nch = x.get_shape()[3].value
        ncl = y.get_shape()[1].value

        # placeholders
        self._input = x
        self._target = y

        # network
        with tf.name_scope('core_network'):
            with tf.variable_scope('33conv64_1'):
                x = conv2d_layer(x, [3, 3], nch,  64)
            with tf.variable_scope('33conv64_2'):
                x = conv2d_layer(x, [3, 3],  64,  64)

            with tf.variable_scope('22maxpool_1'):
                x = max_pool_layer(x, ps=2, ss=2)

            with tf.variable_scope('33conv96_1'):
                x = conv2d_layer(x, [3, 3],  64, 128)
            with tf.variable_scope('33conv96_2'):
                x = conv2d_layer(x, [3, 3], 128, 128)

            with tf.variable_scope('22maxpool_2'):
                x = max_pool_layer(x, ps=2, ss=2)

            with tf.variable_scope('33conv256_1'):
                x = conv2d_layer(x, [3, 3], 128, 256)
            with tf.variable_scope('33conv256_2'):
                x = conv2d_layer(x, [3, 3], 256, 256)

            with tf.variable_scope('22maxpool_3'):
                x = max_pool_layer(x, ps=2, ss=2)

            with tf.variable_scope('33conv512_1'):
                x = conv2d_layer(x, [3, 3], 256, 512)
            with tf.variable_scope('33conv512_2'):
                x = conv2d_layer(x, [3, 3], 512, 512)
            with tf.variable_scope('33conv512_3'):
                x = conv2d_layer(x, [3, 3], 512, 512)

            with tf.variable_scope('22maxpool_4'):
                x = max_pool_layer(x, ps=2, ss=2)

            with tf.variable_scope('33conv512_4'):
                x = conv2d_layer(x, [3, 3], 512, 512)
            with tf.variable_scope('33conv512_5'):
                x = conv2d_layer(x, [3, 3], 512, 512)
            with tf.variable_scope('33conv512_6'):
                x = conv2d_layer(x, [3, 3], 512, 512)

            with tf.variable_scope('22maxpool_5'):
                x = max_pool_layer(x, ps=2, ss=2)

            with tf.variable_scope('fc_1'):
                dim = x.get_shape()[-1]
                x = tf.reshape(x, [-1, dim])
                x = fc_layer(x, dim, dim)
            with tf.variable_scope('fc_2'):
                x = fc_layer(x, dim, dim)
            with tf.variable_scope('fc_3'):
                self._prediction = fc_layer(x, dim, ncl, act=tf.identity)

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


def train_and_test(trn, tst, mbatch_size=50, epochs=20, run_dir='./log/'):
    from math import ceil
    # graph
    x = tf.placeholder(tf.float32, shape=[None, trn[0].shape[1], trn[0].shape[2], trn[0].shape[3]], name='input_layer')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='target')
    model = vggnet(x, y)
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)
    # start session
    sess.run(tf.global_variables_initializer())

    # training
    for epoch in range(epochs):

        for i in range(ceil(trn[0].shape[0] / mbatch_size)):
            s, e = i * mbatch_size, (i + 1) * mbatch_size
            feed_dict = {x: trn[0][s:e, :].astype('float32'), y: trn[1][s:e, :]}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            feed_dict = {x: tst[0][:mbatch_size, :].astype('float32'), y: tst[1][:mbatch_size, :]}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    results = []
    for i in range(ceil(tst[0].shape[0] / mbatch_size)):
        s, e = i * mbatch_size, (i + 1) * mbatch_size
        feed_dict = {x: tst[0][s:e, :].astype('float32'), y: tst[1][s:e, :]}
        results.append(model.evaluation.eval(feed_dict=feed_dict))

    print(results)
    print("Result: " + str(sum(results)/len(results)))


if __name__ == '__main__':
    from util.util import mkrundir
    from util.data import load_mnist
    from util.data import load_cifar10

    LOG_DIR = "./log"

    sess = tf.InteractiveSession()

    print("Train and test on MNIST:")
    DATA_DIR = "~/data/mnist"
    log_dir = mkrundir(LOG_DIR)
    trn, tst = load_mnist(DATA_DIR)
    train_and_test(trn, tst, epochs=1, run_dir=log_dir)

    sess.close()
    # tf.reset_default_graph()
    # sess = tf.InteractiveSession()

    # print("Train and test on CIFAR10:")
    # DATA_DIR = "~/data/cifar10_py"
    # log_dir = mkrundir(LOG_DIR)
    # trn, tst = load_cifar10(DATA_DIR, flatten=False)
    # train_and_test(trn, tst, epochs=200, run_dir=log_dir)

    # sess.close()
