import tensorflow as tf

from util.elements import fc_layer


class linear(object):

    def __init__(self, dim_in, dim_out, lr=0.5):

        # placeholders
        with tf.name_scope('placeholders'):
            self._input = tf.placeholder(tf.float32, shape=[None, dim_in], name='input_layer')
            self._target = tf.placeholder(tf.float32, shape=[None, dim_out], name='target')

        # classifier
        with tf.name_scope('classifier'):
            self._prediction = fc_layer(self._input, dim_in, dim_out, name='output_layer', act=tf.identity)

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target)
            self._loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self._loss)

        # train
        with tf.name_scope('train_op'):
            self._train = tf.train.GradientDescentOptimizer(lr).minimize(self._loss)

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
        return (self._input, self._target)
