import numpy as np
import tensorflow as tf

from .model import rnn


def generate_echo_data(length, batch_size):
    x = np.array(np.random.choice(2, length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)


def train():
    """ Train.

    Parameters:
    -----------

    Return:
    -------

    Raises:
    -------
    """

    model = rnn()
    (x, t) = model.io_placeholders

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
