import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight_variable')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias_variable')


def fully_connected_layer(input_tensor, dim_in, dim_out, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        W = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
        return act(tf.matmul(input_tensor, W) + b)


class fully_connected(object):
    """ """

    def __init__(self, layer_dims):

        # placeholders
        with tf.name_scope('placeholders'):
            self._input = tf.placeholder(tf.float32, shape=[None, layer_dims[0]], name='input_layer')
            self._target = tf.placeholder(tf.float32, shape=[None, layer_dims[-1]], name='target')

        # inference
        with tf.name_scope('core_network'):
            x = self._input
            for i in range(len(layer_dims)-2):
                x = fully_connected_layer(x, layer_dims[i], layer_dims[i+1], 'hidden_layer_'+str(i))

            self._prediction = fully_connected_layer(x, layer_dims[-2], layer_dims[-1],
                                                     'output_layer', act=tf.identity)

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
        return (self._input, self._target)
