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
input_img_size = (256, 256, 3)


class Dataset():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """
    def __init__(self, data_path_train, data_path_test):
        self.train_data = self.apply_preprocess(self.get_data(data_path_train, True, True))
        self.test_data = self.apply_preprocess(self.get_data(data_path_test, False, False)) # dont prprocess later

    def get_data(self, path, shuffle, augment):
        dataset = tf.keras.preprocessing.image_dataset_from_directory(path, image_size=(512, 512), label_mode=None, shuffle=True, class_names=None, batch_size=1)
        return dataset

    def normalize_img(self, img):
        # img = tf.cast(img, dtype=tf.float32)
        return img / 255.0

    def flatten(self, *x):
        return tf.data.Dataset.from_tensor_slices([i for i in x])


    def preprocess_train_image(self, img):
        # Random flip
        img = img[0]
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
        data = data.cache()
        data = data.shuffle(5000)
        data = data.batch(hp.batch_size)
        return data

## to save images in in main

    # RUN THIS IN MAIN TO LOOK AT IMAGES
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
    
    def check_images(self, filepath):
        count = 0
        
        for filename in os.listdir(filepath):
            count +=1
            img = Image.open(filepath + os.sep + filename)
            
            data = open(filepath + os.sep + filename,'rb').read(10)

            # check if file is JPG or JPEG
            if data[:3] == b'\xff\xd8\xff':
                # print(filename+" is: JPG/JPEG.")
                # print("")
                continue
                # break
            # check if file is PNG
            elif data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
                # print(filename+" is: PNG.")
                print(filename + "is png")
                new_filename = filename[:-3]
                
                print(new_filename)
                print(filepath + os.sep + filename, filepath + os.sep +  new_filename + "png")
                os.rename(filepath + os.sep + filename, filepath + os.sep +  new_filename + "png")
            # check if file is GIF
            else:
                print(filename+" is: invalid.")
                
