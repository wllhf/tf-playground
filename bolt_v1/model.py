from __future__ import division

import tensorflow as tf

FILTER_SIZE = [5, 5]
DIM_1L = 32
DIM_2L = 64
DIM_1F = 1024
PS1 = 8
SS1 = 4
PS2 = 8
SS2 = 2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1(x):
    return tf.nn.max_pool(x, ksize=[1, PS1, PS1, 1], strides=[1, SS1, SS1, 1], padding='SAME')


def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1, PS2, PS2, 1], strides=[1, SS2, SS2, 1], padding='SAME')


def inout_placeholders(patch_size, nclasses):
    x = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], patch_size[2]])
    y = tf.placeholder(tf.float32, shape=[None, nclasses])
    return (x, y)


def inference(x, patch_size, nclasses, keep_prob_ph):
    # first convolutional layer
    with tf.name_scope('first_conv'):
        W_conv1 = weight_variable([FILTER_SIZE[0], FILTER_SIZE[1], patch_size[2], DIM_1L])
        b_conv1 = bias_variable([DIM_1L])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_1(h_conv1)

    # second convolutional layer
    with tf.name_scope('second_conv'):
        W_conv2 = weight_variable([FILTER_SIZE[0], FILTER_SIZE[1], DIM_1L, DIM_2L])
        b_conv2 = bias_variable([DIM_2L])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2(h_conv2)

    # first fully connected layer
    with tf.name_scope('first_fully'):
        dim_in = patch_size[0] // (SS1 * SS2) * patch_size[1] // (SS1 * SS2) * DIM_2L
        W_fc1 = weight_variable([dim_in, DIM_1F])
        b_fc1 = bias_variable([DIM_1F])
        h_pool2_flat = tf.reshape(h_pool2, [-1, dim_in])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_ph)

    # output layer
    with tf.name_scope('output'):
        W_fc2 = weight_variable([DIM_1F, nclasses])
        b_fc2 = bias_variable([nclasses])
        y_est = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_est


def loss(y_target, y_est):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_est, y_target))
    return cross_entropy


def train(loss):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_op


def evaluation(y_target, y_est):
    correct_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(y_est, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
