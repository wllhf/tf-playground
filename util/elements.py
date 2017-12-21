import tensorflow as tf


def weight_variable(shape=None, name='weight_variable'):
    initial = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def bias_variable(shape=None, name='bias_variable'):
    initial = tf.constant(0.1, shape=shape) if shape is not None else None
    return tf.get_variable(name=name, initializer=initial)


def fc_layer(input_tensor, dim_in, dim_out, act=tf.nn.relu, dropout=None, name='fc_layer'):
    with tf.name_scope(name):
        W = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
        y = act(tf.matmul(input_tensor, W) + b)
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def fc_decode_layer(input_tensor, act=tf.nn.relu, dropout=None, name='fc_decode_layer'):
    """ Layer shares weights with corresponding fully connected layer in the same variable scope."""
    with tf.name_scope(name):
        W = weight_variable()
        b = bias_variable([W.get_shape()[0]], name='decode_bias')
        y = act(tf.matmul(input_tensor, tf.transpose(W)) + b)
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def conv2d_layer(input_tensor, fshape, nchannels, nfilters,
                 strides=[1, 1, 1, 1], act=tf.nn.relu, dropout=None, padding='SAME', name='conv2d_layer'):

    with tf.name_scope(name):
        W = weight_variable([fshape[0], fshape[1], nchannels, nfilters])
        b = bias_variable([nfilters])
        y = act(tf.nn.conv2d(input_tensor, W, strides=strides, padding=padding) + b)
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def max_pool_layer(input_tensor, ps=2, ss=2, name='max_pool_layer'):
    with tf.name_scope(name):
        return tf.nn.max_pool(input_tensor, ksize=[1, ps, ps, 1], strides=[1, ss, ss, 1], padding='SAME')


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
