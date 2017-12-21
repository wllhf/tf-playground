""" """

import tensorflow as tf

from util.elements import conv2d_layer


class fullyconv(object):
    """
    2015: Fully convolutional network.
    """

    def __init__(self, x, y):
        nch = x.get_shape()[3].value
        ncl = y.get_shape()[1].value

        # placeholders
        self._input = x
        self._target = y

        # network
        with tf.name_scope('core_network'):
