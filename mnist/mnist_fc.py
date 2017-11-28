from math import ceil
import tensorflow as tf

from util.util import mkrundir
from util.data import load_mnist

from shallow.fc import fully_connected

EPOCHS = 60
MINI_BATCH_SIZE = 1000
VAL_SIZE = 3000
TEST_SIZE = 3000
LOG_DIR = "./log"
DATA_DIR = "~/data/mnist"


def main():
    run_dir = mkrundir(LOG_DIR)
    trn, tst = load_mnist(DATA_DIR)

    samples, labels = trn
    nsamples, dim_in = samples.shape[0], samples.shape[1]
    dim_out = labels.shape[1]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # graph
    x = tf.placeholder(tf.float32, shape=[None, dim_in], name='input_layer')
    y = tf.placeholder(tf.float32, shape=[None, dim_out], name='target')

    model = fully_connected(x, y, [2000, 1000, 100])
    (x, y_target, keep_prob) = model.io_placeholder

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :], keep_prob: 0.5}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            feed_dict = {x: tst[0][:TEST_SIZE].astype('float32'), y_target: tst[1][:TEST_SIZE], keep_prob: 1.0}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    feed_dict = {x: tst[0][VAL_SIZE:VAL_SIZE+TEST_SIZE].astype('float32'),
                 y_target: tst[1][VAL_SIZE:VAL_SIZE+TEST_SIZE], keep_prob: 1.0}
    print(model.evaluation.eval(feed_dict=feed_dict))


if __name__ == '__main__':
    main()
