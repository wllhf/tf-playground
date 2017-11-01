import os

import numpy as np

from img_toolbox import label
from img_toolbox import adjustment
from img_toolbox.data.mnist import mnist


def mkrundir(log_dir='./log'):
    run = 0
    while os.path.exists(os.path.join(log_dir, "run%s" % run)):
        run += 1
    run_dir = os.path.join(log_dir, "run%s" % run)
    os.makedirs(run_dir)
    return run_dir


def load_mnist(path, flatten=True, standardize=True):
    # load data
    mnist_obj = mnist(path)
    trn = mnist_obj.train(flatten)
    tst = mnist_obj.test(flatten)

    # adjust data
    samples, labels = trn
    samples = samples if not standardize else adjustment.per_image_standardization(samples)
    samples = samples if flatten else np.expand_dims(samples, axis=3)
    labels = label.one_hot_repr(labels)
    trn = (samples, labels)

    samples, labels = tst
    samples = samples if not standardize else adjustment.per_image_standardization(samples)
    samples = samples if flatten else np.expand_dims(samples, axis=3)
    labels = label.one_hot_repr(labels)
    tst = (samples, labels)

    return trn, tst
