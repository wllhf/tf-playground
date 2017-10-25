import os
from math import ceil
import tensorflow as tf

from img_toolbox import label
from img_toolbox.data.mnist import mnist


class linear(object):
    """ """

    def __init__(self, data, target):

        # inference
        with tf.name_scope('core_network'):
            W = tf.Variable(tf.zeros([int(data.get_shape()[1]), int(target.get_shape()[1])]), name='weight_variable')
            b = tf.Variable(tf.zeros([int(target.get_shape()[1])]), name='bias_variable')
            self._prediction = tf.matmul(data, W) + b

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

        # train
        with tf.name_scope('train_op'):
            self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)

        # evaluation
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(self._prediction, 1))
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


def main():

    EPOCHS = 5
    MINI_BATCH_SIZE = 9000
    LOG_DIR = "./log"

    run = 0
    while os.path.exists(os.path.join(LOG_DIR, "run%s" % run)):
        run += 1
    run_dir = os.path.join(LOG_DIR, "run%s" % run)
    os.makedirs(run_dir)

    # load data
    mnist_obj = mnist("~/data/mnist")
    trn = mnist_obj.train(flatten=True)
    tst = mnist_obj.test(flatten=True)

    samples, labels = trn
    labels = label.one_hot_repr(labels)
    nsamples, dim_in = samples.shape[0], samples.shape[1]
    dim_out = labels.shape[1]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # graph
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, shape=[None, dim_in])
        y_target = tf.placeholder(tf.float32, shape=[None, dim_out])
    model = linear(x, y_target)

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :]}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    labels = label.one_hot_repr(labels)
    print(model.evaluation.eval(feed_dict={x: samples.astype('float32'), y_target: labels}))


if __name__ == '__main__':
    main()
