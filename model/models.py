"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D

import hyperparameters as hp

"""
The GANILLA model, which contains 4 GANs (two discriminators, 2 generators)
"""
class GANILLA(tf.keras.Model):
    def __init__(self):
        super(GANILLA, self).__init__()
        self.num_classes = hp.num_classes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.architecture = [
             
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return loss(labels, predictions)

"""
The loss model, containing modified RESNET blocks.
"""
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.block1 = [
            Conv2D(filters=64, kernel_size=7, strides=1, padding="same", name="block1_conv1"),
            InstanceNormalization(axis=-1),
            Activation('relu') # seems to go after normalization not CONV2d, need to fact check
            MaxPool2D(2, name="block1_conv1"),   
        ]

        #TODO: Upsampling
        self.upsample = []

    def call(self, x):
        """ Passes the image through the network. """

        x = self.block1(x)
        x = self.resnet(x)
        x = self.upsample(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return loss(labels, predictions)
    
    def resnet(inputs, filter_size):
        """ Returns the output of a single resnet block """
        #TODO: Not sure if for each resent block, the filter size decreases so I have it as a variable rn incase it halves.
        # RESNET18 does that but TBH the architecture is a little different from what I see in the paper.
        KERNEL_INIT = RandomNormal(stddev=0.02) 
        KERNEL_SIZE = 3
        
        mod_resnet = [
            Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding='same', 
                kernel_initializer=KERNEL_INIT, name="conv1"),
            InstanceNormalization(axis=-1),
            Activation('relu'),
            MaxPool2d(strides=2),
            Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding='same', 
                kernel_initializer=KERNEL_INIT, name="conv1"),
            InstanceNormalization(axis=-1),
        ]

        output = mod_resent(inputs)
        result = Concatenate()([output, inputs])

        final_layer = [
            Conv2D(filters=filter_size, kernel_size=3, strides(1,1), padding='same', activation='relu', kernel_initializer=KERNEL_INIT)
        ]

        return result



        # vanilla_resnet = [
        #     Conv2D(filters=64, kernel_size=7, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv1"),
        #     MaxPool2D(3, stride=2),
            
        #     Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv2_x"),
        #     Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv2_x"),

        #     Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv3_x"),
        #     Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv3_x"),

        #     Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv4_x"),
        #     Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv4_x"),
            
        #     Conv2D(filters=512, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv4_x"),
        #     Conv2D(filters=512, kernel_size=3, strides=(2,2), padding='same', 
        #         kernel_initializer='random_normal', activation = 'relu', name="conv4_x"),
        #     AveragePooling2D(pool_size=(2,2), padding='same'),
        #     Dense(1000, activation=softmax) 
        # ]

