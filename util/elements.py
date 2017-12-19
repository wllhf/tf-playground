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


def conv2d_layer(input_tensor, fshape, nchannels, nfilters, act=tf.nn.relu, padding='SAME', name='conv2d_layer'):
    with tf.name_scope(name):
        W = weight_variable([fshape[0], fshape[1], nchannels, nfilters])
        b = bias_variable([nfilters])
        return act(tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding=padding) + b)


def max_pool_layer(input_tensor, ps=2, ss=2, name='max_pool_layer'):
    with tf.name_scope(name):
        return tf.nn.max_pool(input_tensor, ksize=[1, ps, ps, 1], strides=[1, ss, ss, 1], padding='SAME')
