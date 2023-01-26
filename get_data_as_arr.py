import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

import tensorflow_datasets as tfds


(ds_train, ds_test) = tfds.load('mnist', split=['train', 'test'], as_supervised=True)


ds_train.batch(100000, drop_remainder=False)
ds_test.batch(100000, drop_remainder=False)


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

data_ite = tfds.as_numpy(ds_train)
images_train = [i[0] for i in data_ite]
labels_train = [i[1] for i in data_ite]

data_ite = tfds.as_numpy(ds_test)
images_test = [i[0] for i in data_ite]
labels_test = [i[1] for i in data_ite]




data_dict = {"images_train" : images_train,
             "images_test"  : images_test,
             "labels_train" : labels_train,
             "labels_test"  : labels_test}

np.savez_compressed("data/mnist_data", **data_dict)
