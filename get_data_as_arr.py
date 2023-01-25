import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

import tensorflow_datasets as tfds
dataset = tfds.load('mnist', split='train', as_supervised=True)
dataset.batch(10, drop_remainder=False)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
arr_4 = np.array(list(tfds.as_numpy(dataset)))[:,0][0][:,:,0]
plt.imshow(arr_4)
plt.show()
#print(np.array(list(tfds.as_numpy(dataset)))[:,1][0])
def get_array(dataset):
    for image, label in tfds.as_numpy(dataset):
        print(image.shape, type(label), label)
        break
"""
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
#print(ds_info)


def normalize_img(image, label):
  '''Normalizes images: `uint8` -> `float32`.'''
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

max_elems = np.iinfo(np.int64).max

ds_train = ds_train.batch(max_elems)

whole_dataset_tensors = tf.data.Dataset.get_single_element(ds_train)

print(type(whole_dataset_tensors[0]))
"""
"""
with tf.compat.v1.Session() as sess:
    whole_dataset_arrays = sess.run(whole_dataset_tensors)
"""
