import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..elements import weight_variable, bias_variable

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


class rnn:
    """ """

    def __init__(self, batch_size, backprop_size, lr=1e-4):
        """ Instatiate RNN model.

        Parameters:
        -----------
          batch_size: int
          backprop_size: int
          lr: float
        """
        self._input = tf.placeholder(tf.float32, [batch_size, backprop_size])
        self._target = tf.placeholder(tf.int32, [batch_size, backprop_size])

        self._Whh = weight_variable()
        self._Wxh = weight_variable()
        self._Why = weight_variable()
        self._h = None

        input_series = tf.unstack(self._input, axis=1)
        label_series = tf.unstack(self._target, axis=1)
        logit_series = []

        # inference
        with tf.name_scope('core_network'):
            for current_batch in input_series:
                self._h = tf.tanh(tf.matmul(self._Whh, self._h) + tf.matmul(self._Wxh, current_batch))
                logit_series.append(tf.matmul(self._Why, self._h))

            self._probabilities = [tf.nn.softmax(l) for l in logit_series]
            self._predictions = [tf.argmax(p, 1) for p in self._probabilities]

        # loss
        with tf.name_scope('loss'):
            cross_entropies = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l, labels=t)
                for l, t in zip(logit_series, label_series)]
            self._loss = tf.reduce_mean(cross_entropies)

        # train
        with tf.name_scope('train_op'):
            self._train = tf.train.AdamOptimizer(lr).minimize(self._loss)

        # evaluation
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._probabilities, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @property
    def prediction(self):
        return (self._predictions, self._probabilities)

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
    def io_placeholders(self):
        return (self._input, self._target)
