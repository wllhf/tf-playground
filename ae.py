import tensorflow as tf

from elements import fc_layer


class ae_classifier(object):
    """ """

    def __init__(self, ae_dims, classifier_dim):

        # placeholders
        with tf.name_scope('placeholders'):
            self._input = tf.placeholder(tf.float32, shape=[None, ae_dims[0]], name='input_layer')
            self._target = tf.placeholder(tf.float32, shape=[None, classifier_dim], name='target')

        # autoencoder
        with tf.name_scope('encode_network'):
            x = fc_layer(self._input, ae_dims[0], ae_dims[1], 'hidden_layer_'+str(1))
            self._latent_space = fc_layer(x, ae_dims[1], ae_dims[2], 'latent_space'+str(1))

        with tf.name_scope('decode_network'):
            x = fc_layer(self._latent_space, ae_dims[2], ae_dims[1], 'hidden_layer_'+str(1))
            self._decoded = fc_layer(x, ae_dims[1], ae_dims[0], 'decoded')

        # classifier
        with tf.name_scope('classifier'):
            self._prediction = fc_layer(self._latent_space, ae_dims[1], classifier_dim, 'output_layer')

        # pre train
        with tf.name_scope('pre_train_op'):
            with tf.name_scope('loss'):
                self._pre_train_loss = tf.reduce_mean(tf.square(self._decoded - self._input))
                tf.summary.scalar('loss', self._pre_train_loss)

            self._pre_train = tf.train.GradientDescentOptimizer(0.5).minimize(self._pre_train_loss)

        # train
        with tf.name_scope('train_op'):
            with tf.name_scope('loss'):
                with tf.name_scope('cross_entropy'):
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction,
                                                                            labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

            self._pre_train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)

        # evaluation
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._prediction, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self._accuracy)

        self._merged = tf.summary.merge_all()

    @property
    def latent_space(self):
        return self._latent_space

    @property
    def prediction(self):
        return self._prediction

    @property
    def loss(self):
        return self._loss

    @property
    def pre_train(self):
        return self._pre_train

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
