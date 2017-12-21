""" """

import tensorflow as tf

from util.elements import res_block


class resnet(object):
    """ Residual network (34 layer version).
    Year         : 2015
    Parameters   :
    Contribution : Skip layers in order to build very deep networks.

    Notes:
    Like all modules in this repository, this is a modified version of the original.
    """

    def __init__(self, x, y):
        nch = x.get_shape()[3].value
        ncl = y.get_shape()[1].value

        # placeholders
        self._input = x
        self._target = y

        # network
        with tf.name_scope('core_network'):
            with tf.variable_scope('map_dim_64'):
                x = res_block(x, [3, 3],  64, 64)
                x = res_block(x, [3, 3],  64, 64)
                x = res_block(x, [3, 3],  64, 64)
            with tf.variable_scope('map_dim_128'):
                x = res_block(x, [3, 3],   64, 128)
                x = res_block(x, [3, 3],  128, 128)
                x = res_block(x, [3, 3],  128, 128)
                x = res_block(x, [3, 3],  128, 128)
            with tf.variable_scope('map_dim_256'):
                x = res_block(x, [3, 3],  128, 256)
                x = res_block(x, [3, 3],  256, 256)
                x = res_block(x, [3, 3],  256, 256)
                x = res_block(x, [3, 3],  256, 256)
                x = res_block(x, [3, 3],  256, 256)
                x = res_block(x, [3, 3],  256, 256)
            with tf.variable_scope('map_dim_512'):
                x = res_block(x, [3, 3],  256, 512)
                x = res_block(x, [3, 3],  512, 512)
                x = res_block(x, [3, 3],  512, 512)
            with tf.variable_scope('fc_layer'):
                dim = x.get_shape()[-1]
                x = tf.reshape(x, [-1, dim])
                self._prediction = fc_layer(x, dim, ncl, act=tf.identity)
