from math import ceil
import tensorflow as tf

from img_toolbox import label
from img_toolbox.data.mnist import mnist


class linear(object):
    """ """

    def __init__(self, data, target):

        # inference
        W = tf.Variable(tf.zeros([int(data.get_shape()[1]), int(target.get_shape()[1])]))
        b = tf.Variable(tf.zeros([int(target.get_shape()[1])]))
        self._prediction = tf.matmul(data, W) + b

        # loss
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=target))

        # train
        self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)

        # evaluation
        correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(self._prediction, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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


def main():

    EPOCHS = 5
    MINI_BATCH_SIZE = 9000

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
    x = tf.placeholder(tf.float32, shape=[None, dim_in])
    y_target = tf.placeholder(tf.float32, shape=[None, dim_out])
    model = linear(x, y_target)

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :]}
            model.train.run(feed_dict=feed_dict)

        if epoch % 1 == 0:
            train_accuracy = model.evaluation.eval(feed_dict=feed_dict)
            print("epoch %d, training accuracy %g" % (epoch+1, train_accuracy))

    # testing
    samples, labels = tst
    labels = label.one_hot_repr(labels)
    print(model.evaluation.eval(feed_dict={x: samples.astype('float32'), y_target: labels}))


if __name__ == '__main__':
    main()
