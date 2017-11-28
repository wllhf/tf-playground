import tensorflow as tf

from util.elements import fc_layer


class sae(object):
    """Implemented as a list of autoencoders, one for each hidden layer, sharing the same variables."""

    def __init__(self, x, dimensions):
        """
        """

        # number of aes equals number of hidden layers
        dimensions.insert(0, x.get_shape()[1].value)
        self._n_aes = len(dimensions)-1
        self._aes = []
        self._loss = []
        self._train = []

        # generate autoencoders
        with tf.name_scope('stacked_autoencoder'):

            for i in range(self._n_aes):
                self._aes.append([x])

                with tf.name_scope('autoencoder_'+str(i)):

                    # encoding layers
                    for l in range(self._n_aes-i):
                        with tf.variable_scope('encode_layer_'+str(l), reuse=(i > 0)):
                            self._aes[i].append(
                                fc_layer(self._aes[i][-1], dimensions[l], dimensions[l+1]))

                    # decoding layers
                    for l in reversed(range(self._n_aes-i)):
                        with tf.variable_scope('decode_layer_'+str(l), reuse=(i > 0)):
                            self._aes[i].append(
                                fc_layer(self._aes[i][-1], dimensions[l+1], dimensions[l]))

        # pre training
        with tf.name_scope('pre_training'):
            with tf.name_scope('loss'):

                for i in range(self._n_aes):
                    self._loss.append(tf.reduce_mean(tf.square(self._aes[i][-1] - self._aes[i][0])))
                    tf.summary.scalar('loss_'+str(i), self._loss[-1])

            with tf.name_scope('training_ops'):

                for i in range(self._n_aes):
                    self._train.append(tf.train.GradientDescentOptimizer(0.5).minimize(self._loss[-1]))

        self._merged = tf.summary.merge_all()

    @property
    def latent_space(self):
        return self._aes[0][self._n_aes]

    @property
    def decoded(self):
        return self._aes[0][-1]

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
        return (self._aes[0][0],)


class sae_classifier(object):
    """ """

    def __init__(self, x, y, ae_dimensions, dim_dropout):

        self._input = x
        self._target = y

        # autoencoder
        self._ae = sae(self._input, ae_dimensions)

        # classifier
        with tf.name_scope('classifier'):
            with tf.variable_scope('fc_layer'):
                self._keep_prob = tf.placeholder(tf.float32)
                fc = fc_layer(self._ae.latent_space, ae_dimensions[-1], dim_dropout, dropout=self._keep_prob)

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
