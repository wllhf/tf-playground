from math import ceil
import tensorflow as tf

from util import mkrundir, load_mnist
from fc import fully_connected

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
    model = fully_connected(dim_in, [1000], dim_out)
    (x, y_target) = model.io_placeholder

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :]}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            feed_dict = {x: tst[0][:TEST_SIZE].astype('float32'), y_target: tst[1][:TEST_SIZE]}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    feed_dict = {x: tst[0][VAL_SIZE:VAL_SIZE+TEST_SIZE].astype('float32'),
                 y_target: tst[1][VAL_SIZE:VAL_SIZE+TEST_SIZE]}
    print(model.evaluation.eval(feed_dict=feed_dict))


if __name__ == '__main__':
    main()
