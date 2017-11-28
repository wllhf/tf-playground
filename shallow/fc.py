import tensorflow as tf

from util.elements import fc_layer


class fully_connected(object):
    """
    Fully connected neural network.

    Fully connected layers \w dropout
    + fully connected output layer \wo dropout
    """

    def __init__(self, x, y, dimensions=[]):
        """
        """

        self._n_layers = len(dimensions)-1

        # placeholders
        self._keep_prob = tf.placeholder(tf.float32)
        self._input = x
        self._target = y

        # network
        self._layers = [self._input]

        with tf.name_scope('network'):
            # hidden layers
            for i in range(self._n_layers):
                with tf.variable_scope('hidden_layer_'+str(i)):
                    self._layers.append(
                        fc_layer(self._layers[-1], self._layers[-1].get_shape()[1].value, dimensions[i],
                                 dropout=self._keep_prob))

            # output layer
            with tf.variable_scope('output_layer'):
                self._layers.append(
                    fc_layer(self._layers[-1],
                             self._layers[-1].get_shape()[1].value, self._target.get_shape()[1].value,
                             act=tf.identity))

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._layers[-1], labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

        # train
        with tf.name_scope('train_op'):
            # self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)
            self._train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # evaluation
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._layers[-1], 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self._accuracy)

        self._merged = tf.summary.merge_all()

    @property
    def prediction(self):
        return self._layers[-1]

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
