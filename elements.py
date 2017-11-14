import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight_variable')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias_variable')


def fc_layer(input_tensor, dim_in, dim_out, name='fc_layer', act=tf.nn.relu):
    with tf.name_scope(name):
        W = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
        return act(tf.matmul(input_tensor, W) + b)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, ps=2, ss=2):
    return tf.nn.max_pool(x, ksize=[1, ps, ps, 1], strides=[1, ss, ss, 1], padding='SAME')
