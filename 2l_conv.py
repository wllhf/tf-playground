import time

import numpy as np
import tensorflow as tf

from img_toolbox import label
from img_toolbox import adjustment
from img_toolbox.data.mnist import mnist

PATCH_SIZE = [28, 28]
NCHANNELS = 1

FILTER_SIZE = [7, 7]
DIM_1L = 32
DIM_2L = 64
DIM_FUL = 1024

EPOCHS = 1000
BATCH_SIZE = 1000


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# load training data
mnist_obj = mnist("~/data/mnist")
trn = mnist_obj.train(flatten=False)
tst = mnist_obj.test(flatten=False)

samples, labels = trn
samples = adjustment.per_image_standardization(samples)
samples = np.expand_dims(samples, axis=3)
labels = label.one_hot_repr(labels)

# stats
n, d = samples.shape[0], samples.shape[1]
nclasses = labels.shape[1]
print("Samples:", n, d, samples.dtype)

# tensorflow session
sess = tf.InteractiveSession()

# input
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE[0], PATCH_SIZE[1], NCHANNELS])
tf.summary.image('input', x, 100)

# output
y_ = tf.placeholder(tf.float32, shape=[None, nclasses])

# first layer
W_conv1 = weight_variable([FILTER_SIZE[0], FILTER_SIZE[1], NCHANNELS, DIM_1L])
b_conv1 = bias_variable([DIM_1L])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([FILTER_SIZE[0], FILTER_SIZE[1], DIM_1L, DIM_2L])
b_conv2 = bias_variable([DIM_2L])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
W_fc1 = weight_variable([PATCH_SIZE[0] // 4 * PATCH_SIZE[1] // 4 * DIM_2L, DIM_FUL])
b_fc1 = bias_variable([DIM_FUL])
h_pool2_flat = tf.reshape(h_pool2, [-1, PATCH_SIZE[0] // 4 * PATCH_SIZE[1] // 4 * DIM_2L])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout (reduce overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = weight_variable([DIM_FUL, nclasses])
b_fc2 = bias_variable([nclasses])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# end nodes
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tensorboard
merged = tf.summary.merge_all()
tb_writer = tf.summary.FileWriter("/tmp/tensorflow/topo/2l_conv/" + str(int(time.time())), sess.graph)

# init
tf.global_variables_initializer().run()

# run
for i in range(EPOCHS):
    i = i % (n // BATCH_SIZE)
    s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
    batch = (samples[s:e, :], labels[s:e, :])

    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0].astype('float32'), y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0].astype('float32'), y_: batch[1], keep_prob: 0.5})
    tb_writer.add_summary(summary, i)

# testing
samples, labels = tst
samples = adjustment.per_image_standardization(samples)
samples = np.expand_dims(samples, axis=3)
labels = label.one_hot_repr(labels)

print("test accuracy %g" % accuracy.eval(feed_dict={x: samples[:1000], y_: labels[:1000], keep_prob: 1.0}))

tb_writer.close()
