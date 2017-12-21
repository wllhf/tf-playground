import pickle

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from img_toolbox import label
from img_toolbox.data.mnist import mnist
from img_toolbox.data.cifar import cifar10


def shuffle_data(x, y):
    indices = np.array(range(y.shape[0]))
    np.random.shuffle(indices)
    return x[indices], y[indices]


def preprocessing(x_train, x_test, per_image_standardize=True, whitening=True):

    processor = ImageDataGenerator(
        samplewise_std_normalization=per_image_standardize,
        zca_whitening=whitening
    )

    processor.fit(x_train)
    x_train = processor.standardize(x_train)
    x_test = processor.standardize(x_test)

    return (x_train, x_test)


def load_mnist(path, flatten=False, standardize=True, whitening=True, shuffle=True, one_hot=True):
    mnist_obj = mnist(path)
    trn = mnist_obj.train(flatten=flatten, integral=False)
    tst = mnist_obj.test(flatten=flatten, integral=False)

    if shuffle:
        trn = shuffle_data(trn[0], trn[1])
        tst = shuffle_data(tst[0], tst[1])

    x_train, y_train = trn
    x_test, y_test = tst

    x_train, x_test = preprocessing(x_train, x_test, standardize, whitening)

    if one_hot:
        y_train = label.one_hot_repr(y_train)
        y_test = label.one_hot_repr(y_test)

    trn = (x_train, y_train)
    tst = (x_test, y_test)

    return trn, tst


def load_cifar10(path, flatten=False, standardize=True, whitening=True, shuffle=True, one_hot=True):
    cifar_obj = cifar10(path)
    trn = cifar_obj.train(flatten=flatten, integral=False)
    tst = cifar_obj.test(flatten=flatten, integral=False)
    print(trn[0].shape, trn[1].shape, trn[0].dtype)

    if shuffle:
        trn = shuffle_data(trn[0], trn[1])
        tst = shuffle_data(tst[0], tst[1])

    x_train, y_train = trn
    x_test, y_test = tst

    x_train, x_test = preprocessing(x_train, x_test, standardize, whitening)

    if one_hot:
        y_train = label.one_hot_repr(y_train)
        y_test = label.one_hot_repr(y_test)

    trn = (x_train, y_train)
    tst = (x_test, y_test)

    return trn, tst


class gaussian(object):
    """ Gaussian distribution. """

    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma

    def sample(self, n):
        return np.random.normal(self._mu, self._sigma, n)


class noisy_uniform(object):
    """ Noisy uniform distribution. """

    def __init__(self, lower_bound, upper_bound):
        self._l = lower_bound
        self._u = upper_bound

    def sample(self, n):
        return np.linspace(self._l, self._u, n) + np.random.random(n) * 0.01
