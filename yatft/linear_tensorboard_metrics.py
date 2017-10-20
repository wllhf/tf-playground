""" TensorBoard visualization.
    run tensorboard --logdir tf-playground/yatft/ and browse shown address

"""

import os
from math import ceil

import tensorflow as tf

from img_toolbox import label
from img_toolbox.data.mnist import mnist


def placeholders(dim_in, dim_out):
    x = tf.placeholder(tf.float32, shape=[None, dim_in])
    y = tf.placeholder(tf.float32, shape=[None, dim_out])
    return (x, y)


def inference(x, dim_in, dim_out):
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([dim_in, dim_out]))
        mean = tf.reduce_mean(W)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(W - mean))))
        tf.summary.scalar('max', tf.reduce_max(W))
        tf.summary.scalar('min', tf.reduce_min(W))
        tf.summary.histogram('histogram', W)

    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([dim_out]))
        mean = tf.reduce_mean(b)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(b - mean))))
        tf.summary.scalar('max', tf.reduce_max(b))
        tf.summary.scalar('min', tf.reduce_min(b))
        tf.summary.histogram('histogram', b)

    with tf.name_scope('Wx_plus_b'):
        y = tf.matmul(x, W) + b
        mean = tf.reduce_mean(y)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(y - mean))))
        tf.summary.scalar('max', tf.reduce_max(y))
        tf.summary.scalar('min', tf.reduce_min(y))
        tf.summary.histogram('histogram', y)

    return y


def loss(y_target, y_est):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_est, labels=y_target))
    tf.summary.scalar('loss', cross_entropy)
    return cross_entropy


def train(loss):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    return train_step


def evaluation(y_target, y_est):
    correct_prediction = tf.equal(tf.argmax(y_target, 1), tf.argmax(y_est, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
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
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)
    merged = tf.summary.merge_all()

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :]}
            _, summary = sess.run([train_op, merged], feed_dict=feed_dict)

        if epoch % 1 == 0:
            _, summary = sess.run([accuracy, merged], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    labels = label.one_hot_repr(labels)
    print(accuracy.eval(feed_dict={x: samples.astype('float32'), y_target: labels}))


if __name__ == '__main__':
    main()
