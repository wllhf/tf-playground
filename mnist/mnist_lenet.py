from math import ceil
import tensorflow as tf

from util.util import mkrundir
from util.data import load_mnist
from cnn.lenet import lenet

EPOCHS = 20
MINI_BATCH_SIZE = 100
LOG_DIR = "./log"
DATA_DIR = "~/data/mnist"

PATCH_SIZE = [28, 28]
NCLASSES = 10
NCHANNELS = 1


def main():
    run_dir = mkrundir(LOG_DIR)
    trn, tst = load_mnist(DATA_DIR, flatten=False)

    samples, labels = trn
    nsamples = samples.shape[0]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # graph
    x = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE[0], PATCH_SIZE[1], NCHANNELS], name='input_layer')
    y = tf.placeholder(tf.float32, shape=[None, NCLASSES], name='target')

    model = lenet(x, y)
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
            feed_dict = {x: samples[:MINI_BATCH_SIZE, :].astype('float32'),
                         y_target: labels[:MINI_BATCH_SIZE, :]}
            _, summary = sess.run([model.evaluation, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    nsamples = samples.shape[0]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # samples, labels = samples[:MINI_BATCH_SIZE], labels[:MINI_BATCH_SIZE]
    results = []
    for i in range(iterations):
        s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
        feed_dict = {x: samples[s:e, :].astype('float32'), y_target: labels[s:e, :]}
        results.append(model.evaluation.eval(feed_dict=feed_dict))

    print(sum(results)/len(results))


if __name__ == '__main__':
    main()
