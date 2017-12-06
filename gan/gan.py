import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util.elements import fc_layer
from util.data import gaussian, noisy_uniform
from util.util import mkrundir


class gan(object):

    def __init__(self, x, z, dim_gen_hidden, dim_dis_hidden):

        self._x = x
        self._z = z

        # generator
        with tf.name_scope('generator') and tf.variable_scope('generator'):
            with tf.variable_scope('l0'):
                gen_l0 = fc_layer(self._z, z.get_shape()[1].value, dim_gen_hidden, act=tf.tanh)
            with tf.variable_scope('l1'):
                gen_l1 = fc_layer(gen_l0, dim_gen_hidden, dim_gen_hidden, act=tf.tanh)
            with tf.variable_scope('l_out'):
                self._G = fc_layer(gen_l1, dim_gen_hidden, x.get_shape()[1].value, act=tf.identity)

        # discriminator
        with tf.name_scope('discriminator') and tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dis_l0'):
                l01 = fc_layer(self._x, x.get_shape()[1].value, dim_dis_hidden, act=tf.tanh)
                l02 = fc_layer(self._G, x.get_shape()[1].value, dim_dis_hidden, act=tf.tanh)
            with tf.variable_scope('dis_l1'):
                l11 = fc_layer(l01, dim_dis_hidden, dim_dis_hidden, act=tf.tanh)
                l12 = fc_layer(l02, dim_dis_hidden, dim_dis_hidden, act=tf.tanh)
            with tf.variable_scope('dis_l2'):
                l21 = fc_layer(l11, dim_dis_hidden, dim_dis_hidden, act=tf.tanh)
                l22 = fc_layer(l12, dim_dis_hidden, dim_dis_hidden, act=tf.tanh)
            with tf.variable_scope('dis_l3'):
                self._D1 = fc_layer(l21, dim_dis_hidden, 1, act=tf.nn.sigmoid)
                self._D2 = fc_layer(l22, dim_dis_hidden, 1, act=tf.nn.sigmoid)

        # loss
        with tf.name_scope('loss'):
            self._loss_d = -tf.reduce_mean(tf.log(self._D1) + tf.log(1. - self._D2))
            self._loss_g = -tf.reduce_mean(tf.log(self._D2))
            self._sum_d = tf.summary.scalar('dis loss', self._loss_d)
            self._sum_g = tf.summary.scalar('gen loss', self._loss_g)

        # train
        with tf.name_scope('train_op'):
            self._train_d = tf.train.AdamOptimizer(1e-4).minimize(self._loss_d)
            self._train_g = tf.train.AdamOptimizer(1e-3).minimize(self._loss_g)

        # evaluation
        # with tf.name_scope('accuracy'):
        #     correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._prediction, 1))
        #     self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     tf.summary.scalar('accuracy', self._accuracy)

    @property
    def prediction(self):
        return self._prediction

    @property
    def G(self):
        return self._G

    @property
    def D(self):
        return self._D1

    @property
    def loss(self):
        return [self._loss_d, self._loss_g]

    @property
    def train_D(self):
        return self._train_d

    @property
    def train_G(self):
        return self._train_g

    @property
    def summary_D(self):
        return self._sum_d

    @property
    def summary_G(self):
        return self._sum_g

    @property
    def io_placeholder(self):
        return (self._input, self._target, self._keep_prob)


def visualize(dist_z, dist_x, model):

    z_in = np.expand_dims(np.linspace(-5, 5, TEST_SIZE), axis=1)
    res = sess.run([model.G], {z: z_in})[0]
    res = np.array([r[0] for r in res])
    rng = (min(-5, res.min()), max(5, res.max()))

    dist_z = np.histogram(dist_z.sample(TEST_SIZE), range=rng, bins=100, density=True)
    dist_x = np.histogram(dist_x.sample(TEST_SIZE), range=rng, bins=100, density=True)
    dist_xe = np.histogram(np.array(res), range=rng, bins=100, density=True)

    x_in = np.expand_dims(np.linspace(rng[0], rng[1], TEST_SIZE), axis=1)
    boundary = sess.run([model.D], {x: x_in})[0]

    plt.cla()
    plt.xlabel('values')
    plt.ylabel('density')
    plt.title('1D GAN')

    # plt.plot(dist_z[1][:-1], dist_z[0], c='k')
    plt.plot(dist_x[1][:-1], dist_x[0], c='b', label='real data')
    plt.plot(dist_xe[1][:-1], dist_xe[0], c='r', label='fake data')
    plt.plot(x_in, boundary, c='g', label='decision boundary')
    plt.legend()
    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    EPOCHS = 10000
    MINI_BATCH_SIZE = 1000
    VAL_SIZE = 3000
    TEST_SIZE = 10000
    LOG_DIR = "./log"
    run_dir = mkrundir(LOG_DIR)

    x_dist = gaussian(2, 0.1)
    z_dist = noisy_uniform(-5, 5)

    x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
    z = tf.placeholder(tf.float32, shape=(None, 1), name='z')

    model = gan(x, z, 100, 100)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    plt.ion()
    plt.show()

    for i in range(EPOCHS):
        x_sample = np.expand_dims(x_dist.sample(MINI_BATCH_SIZE), axis=1)
        z_sample = np.expand_dims(z_dist.sample(MINI_BATCH_SIZE), axis=1)
        feed_dict = {x: x_sample, z: z_sample}
        _, summary = sess.run([model.train_D, model.summary_D], feed_dict=feed_dict)
        file_writer.add_summary(summary, i)

        z_sample = np.expand_dims(z_dist.sample(MINI_BATCH_SIZE), axis=1)
        feed_dict = {z: z_sample}
        _, summary = sess.run([model.train_G, model.summary_G], feed_dict=feed_dict)
        file_writer.add_summary(summary, i)

        if i % 10 == 0:
            # evaluate generator
            visualize(z_dist, x_dist, model)
