from math import ceil
import tensorflow as tf

from util import mkrundir, load_mnist
from cnn import cnn

EPOCHS = 10
MINI_BATCH_SIZE = 1000
LOG_DIR = "./log"
DATA_DIR = "~/data/mnist"


def main():
    run_dir = mkrundir(LOG_DIR)
    trn, tst = load_mnist(DATA_DIR, flatten=False)

    samples, labels = trn
    nsamples = samples.shape[0]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # graph
    model = cnn()
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
            feed_dict = {x: samples[:MINI_BATCH_SIZE, :].astype('float32'),
                         y_target: labels[:MINI_BATCH_SIZE, :], keep_prob: 1.0}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    samples, labels = samples[:MINI_BATCH_SIZE], labels[:MINI_BATCH_SIZE]
    print(model.evaluation.eval(feed_dict={x: samples.astype('float32'), y_target: labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
