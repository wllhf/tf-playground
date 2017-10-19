from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import model

from ...img_toolbox.label import one_hot_repr
from ...img_toolbox.adjustment import per_image_standardization

PATH = "topo_data/set_03/augmented_02"
LOGDIR = "/tmp/tensorflow/topo/bolt_v1/"
MODELDIR = "topo/models/bolt_v1/ckpt"

EPOCHS = 10
BATCH_SIZE = 200


def prepare_set(path_samples, path_labels, nsamples=None):
    samples = np.load(path_samples)
    samples = per_image_standardization(samples)
    labels = np.load(path_labels)
    labels = one_hot_repr(labels)
    indices = np.random.permutation(samples.shape[0])
    indices = indices if nsamples is None else indices[:nsamples]
    return samples[indices, :], labels[indices, :]


def train():

    # data
    train = prepare_set(os.path.join(PATH, "96_bal_patch_train_cropped.npy"),
                        os.path.join(PATH, "96_bal_label_train_cropped.npy"))

    test = prepare_set(os.path.join(PATH, "96_bal_patch_test_cropped.npy"),
                       os.path.join(PATH, "96_bal_label_test_cropped.npy"), 2000)

    nclasses = train[1].max() + 1
    patch_size = train[0].shape[1:4]
    print("samples: %d, shape: (%d, %d, %d)" % (train[0].shape[0], patch_size[0], patch_size[1], patch_size[2]))

    test_batch = (test[0][:100, :], test[1][:100, :])

    with tf.Graph().as_default():
        # placeholders
        x, y_target = model.inout_placeholders(patch_size, nclasses)
        keep_prob = tf.placeholder(tf.float32)

        # model
        y_est = model.inference(x, patch_size, nclasses, keep_prob)
        loss = model.loss(y_target, y_est)
        train_op = model.train(loss)
        acc_train = model.evaluation(y_target, y_est)
        acc_valid = model.evaluation(y_target, y_est)

        # summaries
        loss_str = tf.summary.scalar('loss', loss)
        acc_train_str = tf.summary.scalar('acc_train', acc_train)
        acc_valid_str = tf.summary.scalar('acc_valid', acc_valid)
        summary = tf.summary.merge([loss_str, acc_train_str])

        # tensorflow session
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(LOGDIR + str(int(time.time())), sess.graph)
        sess.run(init_op)

        tf.add_to_collection("x", x)
        tf.add_to_collection("keep_prob", keep_prob)
        tf.add_to_collection("y_est", y_est)
        tf.add_to_collection("loss", loss)
        tf.add_to_collection("train_op", train_op)
        saver.export_meta_graph(os.path.join(MODELDIR, 'model.meta'))

        # run
        for i in range(EPOCHS):
            start_time = time.time()

            idx = i % (train[0].shape[0] // BATCH_SIZE)
            s, e = idx * BATCH_SIZE, (idx + 1) * BATCH_SIZE
            batch = (train[0][s:e, :], train[1][s:e, :])

            feed_dict = {x: batch[0].astype('float32'), y_target: batch[1], keep_prob: 0.5}
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # summary
            if i % 10 == 0:
                print('step %d: loss = %.2f, %.3f sec' % (i, loss_value, time.time() - start_time))

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)

                feed_dict = {x: test_batch[0].astype('float32'), y_target: test_batch[1], keep_prob: 1.0}
                summary_writer.add_summary(acc_valid_str.eval(session=sess, feed_dict=feed_dict), i)

                summary_writer.flush()

            # saver
            if (i + 1) % 100 == 0 or (i + 1) == EPOCHS:
                checkpoint_file = os.path.join(MODELDIR, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i)


train()
