import tensorflow as tf

from elements import fc_layer


class sae(object):
    """ """

    def __init__(self, x, dimensions):

        self._n_aes = len(dimensions)-1
        self._aes = []
        self._loss = []
        self._train = []

        # generate autoencoders
        with tf.name_scope('stacked_autoencoder'):

            for i in range(self._n_aes):
                self._aes.append([x])

                with tf.name_scope('autoencoder_'+str(i)):

                    for l in range(self._n_aes-i):
                        with tf.variable_scope('encode_layer_'+str(l), reuse=(i > 0)):
                            self._aes[i].append(
                                fc_layer(self._aes[i][-1], dimensions[l], dimensions[l+1]))

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
