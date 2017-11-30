import tensorflow as tf

from tflearn.layers import upsample_2d

from util.elements import conv2d_layer, max_pool_layer, fc_layer


def encode(input_tensor, filter_size, nchannels, nfilters, ps_ss=(2, 2)):
    conv = conv2d_layer(input_tensor, filter_size, nchannels, nfilters)
    return max_pool_layer(conv, ps_ss[0], ps_ss[1])


def decode(input_tensor, filter_size, nchannels, nfilters, upsampling_factor):
    conv = conv2d_layer(input_tensor, filter_size, nchannels, nfilters)
    return upsample_2d(conv, (upsampling_factor, upsampling_factor))


class cae(object):
    """Implemented as a list of convolutional aes, one for each hidden layer, sharing the same variables."""

    def __init__(self, x, nfilters=[32], filter_size=[(3, 3)], ps_ss=[(2, 2)]):
        """
        """

        shape = [x.get_shape()[i].value for i in range(len(x.get_shape()))]
        # number of aes equals number of hidden layers
        nfilters.insert(0, shape[3])
        self._n_aes = len(nfilters)-1
        self._aes = []
        self._loss = []
        self._train = []

        # generate autoencoders
        with tf.name_scope('cae') and tf.variable_scope('cae'):
            for i in range(self._n_aes):
                self._aes.append([x])
                with tf.name_scope('ae_'+str(i)):
                    # encoding layers
                    for l in range(self._n_aes-i):
                        with tf.variable_scope('ae_encode_layer_'+str(l), reuse=tf.AUTO_REUSE):
                            self._aes[i].append(
                                encode(self._aes[i][-1], filter_size[l], nfilters[l], nfilters[l+1], ps_ss[l]))
                    # decoding layers
                    for l in reversed(range(self._n_aes-i)):
                        with tf.variable_scope('ae_decode_layer_'+str(l), reuse=tf.AUTO_REUSE):
                            self._aes[i].append(
                                decode(self._aes[i][-1], filter_size[l], nfilters[l+1], nfilters[l], ps_ss[l][1]))

            # pre training
            with tf.name_scope('pre_training'):
                with tf.name_scope('loss'):
                    for i in range(self._n_aes):
                        cropped = tf.image.resize_image_with_crop_or_pad(self._aes[i][-1], shape[1], shape[2])
                        self._loss.append(tf.reduce_mean(tf.square(cropped - self._aes[i][0])))
                        tf.summary.scalar('loss_'+str(i), self._loss[-1])

                with tf.name_scope('training_ops'):
                    for i in range(self._n_aes):
                        self._train.append(tf.train.GradientDescentOptimizer(0.5).minimize(self._loss[-1]))

        self._merged = tf.summary.merge_all()

    @property
    def latent_space(self):
        return self._aes[0][self._n_aes]

    @property
    def decoded(self):
        return self._aes[0][-1]

    @property
    def loss(self):
        return self._loss

    @property
    def train(self):
        return self._train

    @property
    def summary(self):
        return self._merged

    @property
    def io_placeholder(self):
        return (self._aes[0][0],)


class cae_classifier(object):
    """ """

    def __init__(self, x, y, nfilters=[32], filter_size=[(3, 3)], ps_ss=[(2, 2)], dim_dropout=100):

        self._input = x
        self._target = y
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # autoencoder
        self._ae = cae(self._input, nfilters, filter_size, ps_ss)

        # classifier
        with tf.name_scope('classifier') and tf.variable_scope('classifier'):
            print(self._ae.latent_space.get_shape())
            with tf.variable_scope('dropout'):
                flattened = tf.layers.flatten(self._ae.latent_space)
                fc = fc_layer(flattened, flattened.get_shape()[1], dim_dropout, dropout=self._keep_prob)
            with tf.variable_scope('output'):
                self._prediction = fc_layer(fc, dim_dropout, self._target.get_shape()[1].value, act=tf.identity)

            # loss
            with tf.name_scope('loss_ft'):
                with tf.name_scope('cross_entropy'):
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction,
                                                                            labels=self._target)
                self._loss = tf.reduce_mean(cross_entropy)
                sum_loss_ft = tf.summary.scalar('loss_ft', self._loss)

            # train
            with tf.name_scope('train_op_ft'):
                # self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)
                self._train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            # evaluation
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self._target, 1), tf.argmax(self._prediction, 1))
                self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                sum_acc_ft = tf.summary.scalar('accuracy', self._accuracy)

        self._merged = tf.summary.merge([sum_loss_ft, sum_acc_ft])

    @property
    def latent_space(self):
        return self._ae.latent_space

    @property
    def prediction(self):
        return self._prediction

    @property
    def pre_train_loss(self):
        return self._ae.loss

    @property
    def loss(self):
        return self._loss

    @property
    def pre_train(self):
        return self._ae.train

    @property
    def fine_tune(self):
        return self._train

    @property
    def evaluation(self):
        return self._accuracy

    @property
    def summary(self):
        return self._merged

    @property
    def ae_summary(self):
        return self._ae.summary

    @property
    def io_placeholder(self):
        return (self._ae.io_placeholder[0], self._target, self._keep_prob)
