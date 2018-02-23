import numpy as np
import tensorflow as tf

from elements import conv2d_layer


def res_block(input_tensor, fshape, nchannels, nfilters,
              strides=None, act=tf.nn.relu, dropout=None, padding=None, name='res_block'):
    """ Residual block from 'Deep Residual Learning for Image Recognition'.
        https://arxiv.org/abs/1512.03385

    Uses dimension matching strategy 'B' (1x1 convolution).

    TODO: Integrate batch norm.

    Args:
      input_tensor: Input Tensor.
      fshape: List or tuple with filter dimensions.
      nchannels: Last dimension of input_tensor.
      nfilters: Output dimension.
      strides: Not implemented.
      act: Activation function (default: tf.nn.relu).
      dropout: Not implemented.
      padding: Not implemented.
      name: Name of residual block.

    Returns:
      output_tensor
    """
    with tf.name_scope(name):
        x = conv2d_layer(input_tensor, fshape, nchannels, nfilters, act=act, name=name+'_conv1')
        x = conv2d_layer(x, fshape, nfilters, nfilters, act=tf.identity, name=name+'_conv2')

        if nchannels != nfilters:
            input_matched = conv2d_layer(input_tensor, [1, 1], nchannels, nfilters,
                                         act=tf.nn.identity, name=name+'_matching')
            return act(x + input_matched)
        else:
            return act(x + input_tensor)


def res_block_bottleneck(input_tensor, fshape, nchannels, nfilters, nfilters_out,
                         strides=None, act=tf.nn.relu, dropout=None, padding=None, name='res_block'):
    """ Residual block with bottleneck from 'Deep Residual Learning for Image Recognition'.
        https://arxiv.org/abs/1512.03385

    Uses dimension matching strategy 'B' (1x1 convolution).

    TODO: Integrate batch norm.

    Args:
      input_tensor: Input Tensor.
      fshape: List or tuple with filter dimensions.
      nchannels: Last dimension of input_tensor.
      nfilters: Bottleneck dimension.
      nfilters_out: Output dimension.
      strides: Not implemented.
      act: Activation function (default: tf.nn.relu).
      dropout: Not implemented.
      padding: Not implemented.
      name: Name of residual block.

    Returns:
      output_tensor
    """
    with tf.name_scope(name):
        x = conv2d_layer(input_tensor, [1, 1], nchannels, nfilters, act=act, name=name+'_conv1')
        x = conv2d_layer(x, fshape, nfilters, nfilters, act=act, name=name+'_conv2')
        x = conv2d_layer(x, [1, 1], nfilters, nfilters_out, act=tf.identity, name=name+'_conv3')

        if nchannels != nfilters_out:
            input_matched = conv2d_layer(input_tensor, [1, 1], nchannels, nfilters_out,
                                         act=tf.nn.identity, name=name+'_matching')
            return act(x + input_matched)
        else:
            return act(x + input_tensor)
