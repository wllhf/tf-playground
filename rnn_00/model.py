import numpy as np
import tensorflow as tf


def weight_variable(shape=None, name='weight_variable'):
    init = tf.variance_scaling_initializer(
        scale=1.0, mode='fan_out', distribution='uniform', seed=None, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=init)


def bias_variable(shape=None, name='bias_variable'):
    initial = tf.constant(0.1, shape=shape) if shape is not None else None
    return tf.get_variable(name=name, initializer=initial)


class rnn_cell:
    """
    Almost most basic RNN cell. This implementation is supposed for didactic purposes.
    For other purposes it is recommended to use the official tensorflow implementation.

    Args:
      state_size: int, The dimension of the state of the RNN cell.
      target_size: int, The dimension of the output of the RNN cell.
      act: Nonlinearity to use.  Default: `tanh`.
      name: String, the name of the layer.
    """

    def __init__(self, state_size, target_size, act=None, name='rnn_cell'):
        self._state_size = state_size
        self._target_size = target_size
        self._kernel_h = None
        self._kernel_y = None
        self._bias_h = None
        self._bias_y = None
        self._act = act or tf.tanh
        self._name = name
        self._built = False

    def __call__(self, x, h):
        if not self._built:
            self.build(x.get_shape())

        return self.call(x, h)

    def build(self, x_shape):
        if not self._built:
            with tf.variable_scope(self._name+'_h'):
                self._kernel_h = weight_variable(shape=[x_shape[1].value + self._state_size, self._state_size])
                self._bias_h = bias_variable(shape=[self._state_size])
            with tf.variable_scope(self._name+'_y'):
                self._kernel_y = weight_variable(shape=[self._state_size, self._target_size])
                self._bias_y = bias_variable(shape=[self._target_size])
            self._built = True

    def call(self, x, h):
        # h = act(Whh*h + Wxh*x + b)
        state = self._act(tf.nn.bias_add(tf.matmul(tf.concat([x, h], axis=1), self._kernel_h), self._bias_h))
        logit = tf.matmul(h, self._kernel_y) + self._bias_y
        return logit, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros(shape=[batch_size, self._state_size], dtype=dtype)


def dynamic_rnn(cell, inputs, state=None):
    """
    Args:
      cell: RNN cell.
      inputs: Array with n batches of size d (n x d)
      state: State of RNN cell. Default: None

    Return:
      probabilities, logits
    """
    batch_size = inputs.shape[1]
    state = cell.zero_state(batch_size, inputs.dtype) if state is None else state
    logits_series = []
    for inp in tf.unstack(inputs, axis=0):
        inp = tf.reshape(inp, [batch_size, 1])
        out, state = cell(inp, state)
        logits_series.append(out)
    probabilities_series = [tf.nn.softmax(logits) for logits in logits_series]
    prediction_series = [tf.argmax(probs, axis=1) for probs in probabilities_series]
    return prediction_series, probabilities_series, logits_series


def train_rnn(cell, inputs, targets, state=None, lr=1e-4):
    """
    Args:
      cell: RNN cell.
      inputs: Array with n batches of size d (n x d)
      targets: List of expected RNN outputs.
      state: State of RNN cell. Default: None
      lr: float, Learning rate.

    Return:
      op: Training operation.
    """
    _, _, logits_series = dynamic_rnn(cell=cell, inputs=inputs, state=state)
    losses_series = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l, labels=t)
                     for l, t in zip(logits_series, tf.unstack(targets, axis=0))]
    total_loss = tf.reduce_mean(losses_series)
    # train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
    train_op = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
    return train_op, total_loss
