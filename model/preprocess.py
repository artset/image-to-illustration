"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.data_path = data_path

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.zeros((3,))
        self.calc_mean_and_std()

        # Setup data generators
        # TRAIN DATA
        self.train_data = self.get_data(os.path.join(self.data_path, "train/"), True, True)
        # TEST DATA
        # self.test_data = self.get_data(os.path.join(self.data_path, "test/"), False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none
        Returns: none
        """
        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros((hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            
            img = img.resize((hp.img_size, hp.img_size))
            
            img = np.array(img, dtype=np.float32)
            

            print(img.shape, file_path)

            img /= 255.


            data_sample[i] = img

        # Calculate the pixel-wise mean and standard deviation
        # of the images in data_sample and store them in
        # self.mean and self.std respectively.
        # ==========================================================
        self.mean = np.mean(data_sample, axis=(0, 1, 2))
        self.std = np.std(data_sample, axis=(0,1,2))
        # ==========================================================
        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """
        # Standardize the input image. Use self.mean and self.std
        # that were calculated in calc_mean_and_std() to perform
        # the standardization.
        # =============================================================
        img = (img - self.mean) / self.std
        # =============================================================

        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = img / 255.
        # img = self.standardize(img)
        img = tf.image.resize(img, [256, 256], preserve_aspect_ratio=False,antialias=False, name=None)
        return img

    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """
        if augment:
            #       Use the arguments of ImageDataGenerator()
            #       to augment the data. Leave the
            #       preprocessing_function argument as is unless
            #       you have written your own custom preprocessing
            #       function (see custom_preprocess_fn()).
            #
            # Documentation for ImageDataGenerator: https://bit.ly/2wN2EmK
            #
            # ============================================================
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn, 
                horizontal_flip=True, 
                zoom_range=[.5, .8],
                fill_mode='nearest')
            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # model takes 256x256
        img_size = hp.img_size

        # Form image data generator from directory structure
        # data_gen = data_gen.flow_from_directory(
        #     path,
        #     target_size=(img_size, img_size),
        #     save_to_dir=os.path.join(self.data_path, "preprocess/"),
        #     batch_size=hp.batch_size,
        #     shuffle=shuffle)
        file_list=[]
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Import images
        images = []
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = np.array(img)
            img = self.preprocess_fn(img)
            img = np.expand_dims(img, axis=0)
            # img  = tf.keras.preprocessing.image.img_to_array(img, data_format=None, dtype=None)
            images.append(img)
        # print(np.array(images).shape)
            for x, val in zip(data_gen.flow(img, save_to_dir=os.path.join(self.data_path, "train/preprocess/"), save_prefix='aug', save_format='png'), range(10)):
                pass


        
        return data_gen
