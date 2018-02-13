import tensorflow as tf


from ..util.elements import conv2d_layer, max_pool_layer, fc_layer


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


def train_and_test(trn, tst, epochs=50, mbatch_size=100, log_dir="./log"):
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

    model = cnn(x, y)
    (x, y_target, keep_prob) = model.io_placeholder

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    # training
    for epoch in range(epochs):

        for i in range(iterations):
            s, e = i * mbatch_size, (i + 1) * mbatch_size
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :], keep_prob: 0.5}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            feed_dict = {x: samples[:mbatch_size, :].astype('float32'),
                         y_target: labels[:mbatch_size, :], keep_prob: 1.0}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    nsamples = samples.shape[0]
    iterations = ceil(nsamples / mbatch_size)

    # samples, labels = samples[:MBATCH_SIZE], labels[:MBATCH_SIZE]
    results = []
    for i in range(iterations):
        s, e = i * mbatch_size, (i + 1) * mbatch_size
        feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :], keep_prob: 1.0}
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
