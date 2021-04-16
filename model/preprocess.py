import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

import hyperparameters as hp

# Define necessary vals
tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE

orig_img_size = (512, 512)
input_img_size = (1, 256, 256, 3)


class Dataset():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """
    def __init__(self, data_path_train, data_path_test):
        self.train_data = self.apply_preprocess(self.get_data(data_path_train, True, True))
        self.test_data = self.apply_preprocess(self.get_data(data_path_test, False, False)) # dont prprocess later

    def get_data(self, path, shuffle, augment):
        dataset =tf.keras.preprocessing.image_dataset_from_directory(path, image_size=(512, 512), batch_size=hp.batch_size, label_mode=None, shuffle=True, class_names=None)
        return dataset

    def normalize_img(self, img):
        # img = tf.cast(img, dtype=tf.float32)
        return img / 255.0

    def flatten(self, *x):
        return tf.data.Dataset.from_tensor_slices([i for i in x])


    def preprocess_train_image(self, img):
        # Random flip
        img = tf.image.random_flip_left_right(img)
        # Resize to the original size first
        img = tf.image.resize(img, [*orig_img_size])
        # Random crop to 256X256
        other_img = tf.image.random_crop(img, size=[*input_img_size])
        img = tf.image.random_crop(img, size=[*input_img_size])
        # Normalize the pixel values in the range 0-1
        img = self.normalize_img(img)
        other_img = self.normalize_img(other_img)
        return img, other_img

    def apply_preprocess(self, dataset):
        data = (
			dataset.map(self.preprocess_train_image, num_parallel_calls=autotune)
		)
        data = data.flat_map(self.flatten)
        data = data.shuffle(10000)

        return data



## to save images in in main


    # list_data = list(illo_data.train_data.as_numpy_iterator())
    # print(list_data)
    # print("length of data", len(list_data))
    # count = 1
    # for d in list_data:
    #     print("d", d, d.shape)
    #     print("d0", d[0].shape)
    #     # im =  Image.fromarray(d[0])
    #     plt.imshow(d[0])
    #     plt.savefig("figure" + str(count))
    #     count += 1
        # break