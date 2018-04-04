import numpy as np
import tensorflow as tf

from .model import rnn_cell, dynamic_rnn, train_rnn


def generate_echo_data(num_classes, length, echo_delay, batch_size):
    x = np.array(np.random.choice(num_classes, length, p=[0.5, 0.5]))
    y = np.roll(x, echo_delay)
    y[0:echo_delay] = 0
    x = x.reshape((-1, batch_size))
    y = y.reshape((-1, batch_size))
    return (x.astype('float32'), y.astype('int32'))


if __name__ == '__main__':
    epochs = 100
    state_size = 4
    num_classes = 2
    test_length = 30
    series_length = 50000
    echo_delay = 3
    batch_size = 5
    backprop_length = 15

    # rnn cell
    inp_tst = tf.placeholder(shape=[test_length//batch_size, batch_size], dtype='float32')
    inp_trn = tf.placeholder(shape=[backprop_length, batch_size], dtype='float32')
    tgt = tf.placeholder(shape=[backprop_length, batch_size], dtype='int32')
    cell = rnn_cell(state_size, num_classes, act=tf.tanh)
    prediction, _, _ = dynamic_rnn(cell, inp_tst)
    train_op, loss_op = train_rnn(cell, inp_trn, tgt)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        for e in range(epochs):
            x, y = generate_echo_data(num_classes=num_classes,
                                      length=series_length,
                                      echo_delay=echo_delay,
                                      batch_size=batch_size)

            for i in range(x.shape[0]//backprop_length):
                s, e = i*backprop_length, i*backprop_length+backprop_length
                feed_dict = {inp_trn: x[s:e, :], tgt: y[s:e, :]}
                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
                print(loss)

        # testing
        x, y = generate_echo_data(num_classes=num_classes,
                                  length=30,
                                  echo_delay=echo_delay,
                                  batch_size=batch_size)

        feed_dict = {inp_tst: x}
        p = sess.run(prediction, feed_dict)
        print(np.ravel(y))
        print(np.ravel(np.vstack(p)))
