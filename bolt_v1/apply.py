from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import tensorflow as tf

from ...img_toolbox import sample
from ...img_toolbox import adjustment

MODELDIR = "topo/models/bolt_v1/ckpt"


def apply_to_image(img, grid_size=[5, 5]):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(os.path.join(MODELDIR, 'model.meta'))
    new_saver.restore(sess, tf.train.latest_checkpoint(MODELDIR))
    y_est = tf.get_collection('y_est')[0]
    x = tf.get_collection('x')[0]
    keep_prob = tf.get_collection('keep_prob')[0]

    patch_size = tuple(x.get_shape().as_list()[1:3])
    coords = sample.grid_sample_coords(img.shape, grid_size=grid_size, max_patch_size=patch_size)
    patches = sample.at(img, coords, patch_size=patch_size, flatten=False)
    patches = adjustment.per_image_standardization(patches)

    feed_dict = {x: patches.astype('float32'), keep_prob: 1.0}
    result = np.squeeze(np.array(sess.run([y_est], feed_dict=feed_dict)))
    result_norm = (result - result.min()) / (result.max() - result.min())
    result_norm[result[:, 0] > result[:, 1], 1] = 0
    result_norm[result_norm[:, 1] < 0.8, 1] = 0
    # print(result.min(), result.max())

    img_res = np.zeros(img.shape[:2])
    img_res[coords[:, 0], coords[:, 1]] = result_norm[:, 1]

    plt.subplot(121)
    plt.imshow(img, interpolation='None')
    plt.subplot(122)
    plt.imshow(img_res, cmap="hot", interpolation='None')
    plt.colorbar()
    plt.show()

    return result


img = skimage.io.imread("/home/mw/Workspace/topo_data/set_02_falc/test_image.png")[:, :, :3]
apply_to_image(img)
