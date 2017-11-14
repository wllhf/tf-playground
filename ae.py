import tensorflow as tf

from elements import fc_layer


class ae(object):
    """ """

    def __init__(self, dim_in, dim_latent):

        # placeholders
        with tf.name_scope('placeholders'):
            self._input = tf.placeholder(tf.float32, shape=[None, dim_in], name='input_layer')

        # autoencoder
        with tf.name_scope('autoencoder'):
            self._latent_space = fc_layer(self._input, dim_in, dim_latent, 'encode_layer')
            self._decoded = fc_layer(self._latent_space, dim_latent, dim_in, 'decode_layer')

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

    def __init__(self, dim_in, dim_latent, dim_classifier):

        # placeholders
        with tf.name_scope('placeholders'):
            self._target = tf.placeholder(tf.float32, shape=[None, dim_classifier], name='target')

        # autoencoder
        self._ae = ae(dim_in, dim_latent)

        # classifier
        with tf.name_scope('classifier'):
            self._prediction = fc_layer(self._ae.latent_space, dim_latent, dim_classifier,
                                        name='output_layer', act=tf.identity)

        # loss
        with tf.name_scope('loss_ft'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            sum_loss_ft = tf.summary.scalar('loss_ft', self._loss)

        # train
        with tf.name_scope('train_op_ft'):
            self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)

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
        return (self._ae.io_placeholder[0], self._target)
