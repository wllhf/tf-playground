import tensorflow as tf

from elements import fc_layer


class fully_connected(object):
    """ """

    def __init__(self, dim_in, dim_hidden, dim_out):

        # placeholders
        with tf.name_scope('placeholders'):
            self._input = [tf.placeholder(tf.float32, shape=[None, dim_in], name='input_layer')]
            self._target = tf.placeholder(tf.float32, shape=[None, dim_out], name='target')

        # inference
        with tf.name_scope('core_network'):
            if dim_hidden:
                self._input.append(fc_layer(self._input[-1], dim_in, dim_hidden[0], 'hidden_layer_0'))

                for i in range(len(dim_hidden)-1):
                    self._input.append(
                        fc_layer(self._input[-1], dim_hidden[i], dim_hidden[i+1], 'hidden_layer_'+str(i)))

                self._prediction = fc_layer(self._input[-1], dim_hidden[-1], dim_out,
                                            name='output_layer', act=tf.identity)

            else:
                self._prediction = fc_layer(self._input[-1], dim_in, dim_out,
                                            name='output_layer', act=tf.identity)

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

        # train
        with tf.name_scope('train_op'):
            self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)

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
        return (self._input[0], self._target)
