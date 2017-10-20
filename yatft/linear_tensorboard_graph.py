""" TensorBoard visualization.
    run tensorboard --logdir tf-playground/yatft/ and browse shown address

"""

import os
from math import ceil

import tensorflow as tf

from img_toolbox import label
from img_toolbox.data.mnist import mnist


def placeholders(dim_in, dim_out):
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, shape=[None, dim_in], name='input')
        y = tf.placeholder(tf.float32, shape=[None, dim_out], name='target')
    return (x, y)


def inference(x, dim_in, dim_out):
    with tf.name_scope('core_network'):
        W = tf.Variable(tf.zeros([dim_in, dim_out]), name='weight_variable')
        b = tf.Variable(tf.zeros([dim_out]), name='bias_variable')
        y = tf.matmul(x, W) + b
    return y


def loss(y_target, y_est):
    with tf.name_scope('loss'):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_est, labels=y_target)
        cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def train(loss):
    with tf.name_scope('train_op'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    return train_step


def evaluation(y_target, y_est):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(y_est, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def main():

    EPOCHS = 5
    MINI_BATCH_SIZE = 9000
    LOG_DIR = "./log"

    run = 0
    while os.path.exists(os.path.join(LOG_DIR, "run%s" % run)):
        run += 1
    run_dir = os.path.join(LOG_DIR, "run%s" % run)
    os.makedirs(run_dir)

    # load data
    mnist_obj = mnist("~/data/mnist")
    trn = mnist_obj.train(flatten=True)
    tst = mnist_obj.test(flatten=True)

    samples, labels = trn
    labels = label.one_hot_repr(labels)
    nsamples, dim_in = samples.shape[0], samples.shape[1]
    dim_out = labels.shape[1]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # build compute graph
    (x, y_target) = placeholders(dim_in, dim_out)
    y_est = inference(x, dim_in, dim_out)
    loss_op = loss(y_target, y_est)
    train_op = train(loss_op)
    accuracy = evaluation(y_target, y_est)

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :]}
            train_op.run(feed_dict=feed_dict)

        if epoch % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("epoch %d, training accuracy %g" % (epoch+1, train_accuracy))

    # testing
    samples, labels = tst
    labels = label.one_hot_repr(labels)
    print(accuracy.eval(feed_dict={x: samples.astype('float32'), y_target: labels}))


if __name__ == '__main__':
    main()
