import tensorflow as tf


def weight_variable(shape=None, name='weight_variable'):
    # init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    init = tf.variance_scaling_initializer(
        scale=1.0, mode='fan_out', distribution='uniform', seed=None, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=init)


def bias_variable(shape=None, name='bias_variable'):
    initial = tf.constant(0.1, shape=shape) if shape is not None else None
    return tf.get_variable(name=name, initializer=initial)


def batch_norm(input_tensor, is_training=False, decay=0.999, variance_epsilon=0.001):
    shape = input_tensor.get_shape()
    mean = tf.get_variable(name='mean_variable', initializer=tf.constant(0.0, shape=[shape[-1]]), trainable=False)
    var = tf.get_variable(name='var_variable', initializer=tf.constant(0.0, shape=[shape[-1]]), trainable=False)
    beta = tf.get_variable(name='beta_variable', initializer=tf.constant(0.0, shape=[shape[-1]]))
    gamma = tf.get_variable(name='gamma_variable', initializer=tf.constant(1.0, shape=[shape[-1]]))

    if is_training:
        batch_mean, batch_var = tf.nn.moments(input_tensor, [i for i in range(len(shape)-1)], keep_dims=False)
        train_mean = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(var, var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, gamma, variance_epsilon)
    else:
        return tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, variance_epsilon)


def fc_layer(input_tensor, dim_in, dim_out, act=tf.nn.relu, dropout=None, name='fc_layer'):
    with tf.name_scope(name):
        W = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
        y = act(tf.matmul(input_tensor, W) + b)
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def fc_layer_bn(input_tensor, dim_in, dim_out, act=tf.nn.relu, dropout=None, training=False, name='fc_layer'):
    with tf.name_scope(name):
        W = weight_variable([dim_in, dim_out])
        z = tf.matmul(input_tensor, W)
        y = act(batch_norm(z, training))
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def fc_decode_layer(input_tensor, act=tf.nn.relu, dropout=None, name='fc_decode_layer'):
    """ Layer shares weights with corresponding fully connected layer in the same variable scope."""
    with tf.name_scope(name):
        W = weight_variable()
        b = bias_variable([W.get_shape()[0]], name='decode_bias')
        y = act(tf.matmul(input_tensor, tf.transpose(W)) + b)
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def conv2d_layer(input_tensor, fshape, nchannels, nfilters,
                 strides=[1, 1, 1, 1], act=tf.nn.relu, dropout=None,
                 padding='SAME', name='conv2d_layer'):

    with tf.variable_scope(name):
        W = weight_variable([fshape[0], fshape[1], nchannels, nfilters])
        b = bias_variable([nfilters])
        y = act(tf.nn.conv2d(input_tensor, W, strides=strides, padding=padding) + b)
        return tf.nn.dropout(y, dropout) if dropout is not None else y


def conv2d_layer_bn(input_tensor, fshape, nchannels, nfilters, strides=[1, 1, 1, 1],
                    act=tf.nn.relu, dropout=None, batchnorm=True, training=False,
                    padding='SAME', name='conv2d_layer_bn'):

    with tf.variable_scope(name):
        W = weight_variable([fshape[0], fshape[1], nchannels, nfilters])
        z = tf.nn.conv2d(input_tensor, W, strides=strides, padding=padding)
        y = act(batch_norm(z, training)) if batchnorm else act(z+bias_variable([nfilters]))
        y = tf.nn.dropout(y, dropout) if dropout is not None else y

    return y


def max_pool_layer(input_tensor, ps=2, ss=2, name='max_pool_layer'):
    with tf.name_scope(name):
        return tf.nn.max_pool(input_tensor, ksize=[1, ps, ps, 1], strides=[1, ss, ss, 1], padding='SAME')
