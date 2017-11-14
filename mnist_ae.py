from math import ceil
import tensorflow as tf

from util import mkrundir, load_mnist
from ae import ae

EPOCHS = 60
MINI_BATCH_SIZE = 1000
LOG_DIR = "./log"
DATA_DIR = "~/data/mnist"


def main():
    run_dir = mkrundir(LOG_DIR)
    trn, tst = load_mnist(DATA_DIR)
    print(trn[0].max(), trn[0].min())

    samples, labels = trn
    nsamples, dim_in = samples.shape[0], samples.shape[1]
    dim_out = labels.shape[1]
    iterations = ceil(nsamples / MINI_BATCH_SIZE)

    # graph
    model = ae(dim_in, 2000)
    (x,) = model.io_placeholder

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(run_dir, sess.graph)

    # training
    for epoch in range(EPOCHS):

        for i in range(iterations):
            s, e = i * MINI_BATCH_SIZE, (i + 1) * MINI_BATCH_SIZE
            feed_dict = {x: samples[s:e, :].astype('float32')}
            _, summary = sess.run([model.train, model.summary], feed_dict=feed_dict)

        if epoch % 1 == 0:
            _, summary = sess.run([model.loss, model.summary], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

    # testing
    samples, labels = tst
    samples, labels = samples[:MINI_BATCH_SIZE], labels[:MINI_BATCH_SIZE]
    print(model.loss.eval(feed_dict={x: samples.astype('float32')}))


if __name__ == '__main__':
    main()
