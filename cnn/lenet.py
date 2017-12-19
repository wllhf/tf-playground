""" http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf """

from math import ceil

import tensorflow as tf

from util.elements import conv2d_layer, max_pool_layer, fc_layer


class lenet(object):
    """
    Simplified version of LeNet-5. Cross entropy on logits instead
    of RBFs. All feature maps between S2 and C3 connected. First
    convolution is padded to use network on unadjusted MNIST data.
    S4 uses variable filter size to adjust to different input dimensions.
    """

    def __init__(self, x, y, linear_conv=False):
        """
        linear_conv: boolean, set True for linear convolutional layers like in the original paper.
        """

        nchannels = x.get_shape()[3].value
        nclasses = y.get_shape()[1].value

        # placeholders
        self._input = x
        self._target = y

        conv_act = tf.identity if linear_conv else tf.nn.relu

        # inference
        with tf.name_scope('core_network'):
            with tf.variable_scope('C1'):
                c1 = conv2d_layer(self._input, [5, 5], nchannels, 6, act=conv_act)
            with tf.variable_scope('S2'):
                s2 = max_pool_layer(c1, ps=2, ss=2)
            with tf.variable_scope('C3'):
                c3 = conv2d_layer(s2, [5, 5], 6, 16, padding='VALID', act=conv_act)
            with tf.variable_scope('S4'):
                s4 = max_pool_layer(c3, ps=2, ss=2)
            with tf.variable_scope('C5'):
                fsize = s4.get_shape()[1:3]
                c5 = conv2d_layer(s4, fsize, 16, 120, padding='VALID', act=conv_act)
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


def train_and_test(trn, tst, mbatch_size=100, epochs=20, run_dir='./log/'):
    # graph
    x = tf.placeholder(tf.float32, shape=[None, trn[0].shape[1], trn[0].shape[2], trn[0].shape[3]], name='input_layer')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='target')
    model = lenet(x, y)
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
            feed_dict = {x: trn[0][:mbatch_size, :].astype('float32'), y: trn[1][:mbatch_size, :]}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    results = []
    for i in range(ceil(tst[0].shape[0] / mbatch_size)):
        s, e = i * mbatch_size, (i + 1) * mbatch_size
        feed_dict = {x: tst[0][s:e, :].astype('float32'), y: tst[1][s:e, :]}
        results.append(model.evaluation.eval(feed_dict=feed_dict))

    print("Result: " + str(sum(results)/len(results)))


if __name__ == '__main__':
    from util.util import mkrundir
    from util.data import load_mnist
    from util.data import load_cifar10

    LOG_DIR = "./log"
    log_dir = mkrundir(LOG_DIR)

    sess = tf.InteractiveSession()

    print("Train and test on MNIST:")
    DATA_DIR = "~/data/mnist"
    trn, tst = load_mnist(DATA_DIR, flatten=False)
    print(trn[0].shape, trn[1].shape)
    train_and_test(trn, tst, epochs=20, run_dir=log_dir)

    sess.close()
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    print("Train and test on CIFAR10:")
    DATA_DIR = "~/data/cifar10_py"
    trn, tst = load_cifar10(DATA_DIR, flatten=False)
    print(trn[0].shape, trn[1].shape)
    train_and_test(trn, tst, epochs=20, run_dir=log_dir)

    sess.close()
