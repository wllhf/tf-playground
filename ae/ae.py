import tensorflow as tf

from util.elements import fc_layer, fc_decode_layer


class ae(object):
    """ """

    def __init__(self, x, dim_latent):

        # placeholder
        self._input = x

        # autoencoder
        with tf.variable_scope('ae', reuse=tf.AUTO_REUSE):
            self._latent_space = fc_layer(self._input, x.get_shape()[1].value, dim_latent)
            self._decoded = fc_decode_layer(self._latent_space)

        # loss
        with tf.name_scope('loss_pt'):
            self._loss = tf.reduce_mean(tf.square(self._decoded - self._input))
            tf.summary.scalar('loss_pt', self._loss)

        # train
        with tf.name_scope('train_op_pt'):
            self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)

        self._merged = tf.summary.merge_all()

    @property
    def latent_space(self):
        return self._latent_space

    @property
    def decoded(self):
        return self._decoded

    @property
    def loss(self):
        return self._loss

    @property
    def train(self):
        return self._train

    @property
    def summary(self):
        return self._merged

    @property
    def io_placeholder(self):
        return (self._input,)


class ae_classifier(object):
    """ """

    def __init__(self, x, y, dim_latent, dim_dropout):

        self._input = x
        self._target = y

        # autoencoder
        self._ae = ae(x, dim_latent)

        # classifier
        with tf.name_scope('classifier'):
            with tf.variable_scope('fc_layer'):
                self._keep_prob = tf.placeholder(tf.float32)
                fc = fc_layer(self._ae.latent_space, dim_latent, dim_dropout, dropout=self._keep_prob)

            with tf.variable_scope('output_layer'):
                self._prediction = fc_layer(fc, dim_dropout, self._target.get_shape()[1].value, act=tf.identity)

        # loss
        with tf.name_scope('loss_ft'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            sum_loss_ft = tf.summary.scalar('loss_ft', self._loss)

        # train
        with tf.name_scope('train_op_ft'):
            # self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)
            self._train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # evaluation
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._prediction, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sum_acc_ft = tf.summary.scalar('accuracy', self._accuracy)

        self._merged = tf.summary.merge([sum_loss_ft, sum_acc_ft])

    @property
    def latent_space(self):
        return self._ae.latent_space

    @property
    def prediction(self):
        return self._prediction

    @property
    def pre_train_loss(self):
        return self._ae.loss

    @property
    def loss(self):
        return self._loss

    @property
    def pre_train(self):
        return self._ae.train

    @property
    def fine_tune(self):
        return self._train

    @property
    def evaluation(self):
        return self._accuracy

    @property
    def summary(self):
        return self._merged

    @property
    def ae_summary(self):
        return self._ae.summary

    @property
    def io_placeholder(self):
        return (self._ae.io_placeholder[0], self._target, self._keep_prob)
