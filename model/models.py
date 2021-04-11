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

        # TODO: Instantiate the 2 pairs of Discrim/Gen
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
The Generator model, containing modified RESNET blocks.
"""
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.block1 = [
            Conv2D(filters=64, kernel_size=7, strides=1, padding="same", name="block1_conv1"),
            InstanceNormalization(axis=-1),
            Activation("relu") # seems to go after normalization not CONV2D, need to fact check if this is legit
            MaxPool2D(2, name="block1_conv1"),   
        ]

        #TODO: Upsampling
        self.upsample = []
        # use  tf.image.resize(image, size=[5,7], method="nearest") for upsampling, unsure about shape

    def call(self, x):
        """ Passes the image through the network. """

        x = self.block1(x)

        #TODO: a guess for the 4 downsampling blocks: call resnet on on 64, 128, 256 , 512
        x = self.resnet(x, 64)
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

        #NOTE: Not totally sure of the strides, this line is a bit confusing: 
        # "We halve feature map size in each layer except Layer-I using convolutions with stride of 2."
        KERNEL_INIT = RandomNormal(stddev=0.02) 
        KERNEL_SIZE = 3
        
        mod_resnet = [
            Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding="same", 
                kernel_initializer=KERNEL_INIT, name="conv1"),
            InstanceNormalization(axis=-1),
            Activation("relu"),
            MaxPool2d(strides=2),
            Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding="same", 
                kernel_initializer=KERNEL_INIT, name="conv1"),
            InstanceNormalization(axis=-1),
        ]
        
        output = inputs
        for l in mod_resnet:
            output = mod_resent(output)

        result = Concatenate()([output, inputs])

        final_layer = [
            Conv2D(filters=filter_size, kernel_size=3, strides(1,1), padding="same", activation="relu", kernel_initializer=KERNEL_INIT)
        ]

        return result


        # vanilla_resnet = [
        #     Conv2D(filters=64, kernel_size=7, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv1"),
        #     MaxPool2D(3, stride=2),
            
        #     Conv2D(filters=64, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv2_x"),
        #     Conv2D(filters=64, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv2_x"),

        #     Conv2D(filters=128, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv3_x"),
        #     Conv2D(filters=128, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv3_x"),

        #     Conv2D(filters=256, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv4_x"),
        #     Conv2D(filters=256, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv4_x"),
            
        #     Conv2D(filters=512, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv4_x"),
        #     Conv2D(filters=512, kernel_size=3, strides=(2,2), padding="same", 
        #         kernel_initializer="random_normal", activation = "relu", name="conv4_x"),
        #     AveragePooling2D(pool_size=(2,2), padding="same"),
        #     Dense(1000, activation=softmax) 
        # ]



"""
The Discriminator model, leveraging PatchGAN architecture.
Determines if the given image is generated or real.

Nice explanation of PatchGAN first bit: https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207
Note that this explanation has only one layer in each block. This may be worthwhile simplification to improve runtime.

Tutorial: https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
"""
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        KERNEL_SIZE = 3
        
        #TODO: Model, this has not been started really.
        self.layers = [
            Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block1_conv1", activation="relu"),
            Conv2D(filters=64, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block1_conv2", activation="relu"),

            Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block2_conv1", activation="relu"),
            Conv2D(filters=128, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block2_conv2", activation="relu"),

            Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block3_conv1", activation="relu"),
            Conv2D(filters=256, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block3_conv2", activation="relu"),

        ]


    def call(self, x):
        x = self.layers(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        # TODO: Select a loss function for your network (see the documentation
        #       for tf.keras.losses)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        return loss(labels, predictions)
