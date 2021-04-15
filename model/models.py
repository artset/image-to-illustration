"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D, BatchNormalization, \
    ZeroPadding2D, Conv2DTranspose, UpSampling2D, Concatenate, LeakyReLU, ReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.lossees import MeanAbsoluteError

import hyperparameters as hp

"""
The GANILLA model, which contains 4 GANs (two discriminators, 2 generators)
"""
class GANILLA(tf.keras.Model):
    def __init__(self):
        super(GANILLA, self).__init__()
        self.num_classes = hp.num_classes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.g1 = Generator("toIllo")
        self.g2 = Generator("toPhoto")
        self.d1 = Discriminator("isIllo")
        self.d2 = Discriminator("isPhoto")
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5

        self.cy_loss = MeanAbsoluteError()
        self.id_loss = MeanAbsoluteError()

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def cycle_loss(real, cycled):
        """
        Gives our model the property such that
        input photo -> illustrator generator -> generated illustration -> photo generator -> output photo
        input illustrator -> photo generator -> generated photo -> illustrator generator -> output illustration
        
        We want i/o photo and i/o illustrator to look the same.

        Note: I was concerned this will be automatically called in the pipeline for .compile() so I renamed it. - KS
        """
        # loss1 = tf.reduce_mean(tf.abs(real - cycled)) # Commented out for a consistent style with the identity loss. -KS
        return self_lambda_cycle * self.cy_loss(real, cycled)

    @staticmethod
    """
        Gives our model the property that generators will do this
        illustrator -> illustrator generator -> illustrator
        photo -> photo generator -> photo
    """
    def identity_loss(real, cycled):
        return self.lambda_identity * self_lambda_cycle * self.id_loss(real, cycled)

    def generate_images(images):
        """
            Evaluate our model
            images || tensor of shape (batch size x 256 x 256 x 3)
        """
        generated = self.g1(images)
        #TODO: depending on if we're doing [-1,1] or [0,1], transform back to [0, 255]
        return generated


    def train_step(input_data):
        """
        input_data || tuple
        Ideally this should be something that looks like (photos, illustration), aka (source, target)
        in the model.fit() we can pass tf.data.Dataset.zip((photos, illustrations)).
        (Source: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)

        returns a dict mapping each loss to its value

        NOTES:
        My guess is that we have to do an actual train() function rather than just compute a single loss function because
        we need more fine grained control over what gradients are updated, and since this model is composed of multiple losses.
        I'm not sure what the consequence would be if we tried to throw this in a single loss function.
        -KS
        """
        photo, illo = input_data
        with tf.GradientTape(persistent=True) as tape:
            # Call Generators
            fake_illos = self.g1(photos)
            fake_photos = self.g2(illos)

            # Call Discriminators
            disc_fake_illos = self.d1(fake_illos)
            disc_real_illos = self.d1(illos)
            disc_fake_photos = self.d2(fake_photos)
            disc_real_photos = self.d2(photos)

            # Adversarial loss
            ad_illos_loss = self.g1.loss_fn(illos)
            ad_photos_loss = self.g2.loss_fn(photos)

            # Discriminator Loss
            disc_illos_loss = self.d1.loss_fn(disc_fake_illos, disc_real_illos)
            disc_photos_loss = self.d2.loss_fn(disc_fake_photos, disc_real_photos)

            # Compute cyclic loss
            cycle_photos = self.g2(fake_illos)
            cycle_illos = self.g1(fake_photos)
            cycle_photos_loss = cycle_loss(photos, cycle_photos)
            cycle_illos_loss = cycle_loss(illos, cycle_illos)

            # Compute identity losses
            same_illos = self.g1(illos)
            same_photos = self.g2(photos)
            id_photos_loss = identity_loss(photos, same_photos)
            id_illos_loss = identity_loss(illos, same_illos)

            # Generator loss: adversarial + cylic + identity
            gen_illos_loss = ad_illos_loss + cycle_illos_loss + id_illos_loss
            gen_photos_loss = ad_photo_loss + cycle_photos_loss + id_photos_loss

        # Compute gradients for generators and discriminators
        grads_g1 = tape.gradient(gen_illos_loss, self.g1.trainable_variables)
        grads_g2 = tape.gradient(gen_photos_loss, self.g2.trainable_variables)
        grads_d1 = tape.gradient(disc_illos_loss, self.d1.trainable_variables)
        grads_d2 = tape.gradient(disc_photos_loss, self.d2.trainable_variables)

        # Apply gradients to generators and discriminators
        self.g1.optimizer.apply_gradients(zip(grads_g1, self.g1.trainable_variables))
        self.g2.optimizer.apply_gradients(zip(grads_g2, self.g2.trainable_variables))
        self.d1.optimizer.apply_gradients(zip(grads_d1, self.d1.trainable_variables))
        self.d2.optimizer.apply_gradients(zip(grads_d2, self.d2.trainable_variables))

        # Return a dict mapping metric names to current value
        return {
            "gen_illos_loss": gen_illos_loss,
            "gen_photos_loss": gen_photos_loss,
            "disc_illos_loss": disc_illos_loss,
            "disc_photos_loss": disc_photos_loss,
        }


"""
The Generator model, containing modified RESNET blocks.
"""
class Generator(tf.keras.Model):
    def __init__(self, name=None):
        super(Generator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        GAMMA_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        self.name = name


        self.layers = [
            # Original model seems to have a reflection pad, but not sure why
            # Block 1
            Conv2D(filters=64, kernel_size=7, strides=1, padding="same", name="block1_conv1"),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
            ReLU() # seems to go after normalization not CONV2D, need to fact check if this is legit
            # MaxPool2D(2, name="block1_conv1"),   
        ]

        self.pre_upsample = [
            # Not sure about filters, strides, padding, or output padding here
            Conv2DTranspose(filters=512, kernel_size=(1,1), padding="same", outpadding=1)
        ]

        self.post_upsample = {
            # Not sure about filters, , padding, or output padding here
            Conv2DTranspose(filters=64, kernel_size=(1,1), padding="same", outpadding=1),
            Conv2DTranspose(filters=64, kernel_size=(7,7), padding="same", outpadding=1)
        }

    def call(self, x):
        """ Passes the image through the network. """
        # TODO: the TAN BLOCKS between skip connections appear to be conv layers needed to properly resize things so that they can be concatenated or summed togehter!!!!
        # Need to add this to the resnet structure overal ^^
        for layer in self.layer:
            x = layer(x)

        #TODO: a guess for the 4 downsampling blocks: call resnet on on 64, 128, 256 , 512
        # Original code seems do downsample twice -- currently upsampling and downsampling twice to match the diagram on page 5
        # Saving intermediate outputs for use in the upsampling skip connections
        layer_1_out = self.resnet(x, 64)
        layer_2_out = self.resnet(layer_1_out, 128)
        layer_3_out = self.resnet(layer_2_out, 256)
        x = self.resnet(layer_3_out, 512)

        for layer in self.pre_upsample:
            x = layer(x)

        # Original code seems to upsample twice 
        x = self.upsample(x, layer_3_out)
        x = self.upsample(x, layer_2_out)
        x = self.upsample(x, layer_1_out)
        # Not sure about the size
        tf.image.resize(image, size=[5,7], method="nearest")
        
        for layer in self.post_upsample(x):
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(disc_real_output):
        """ Loss function for model. """
        bce = tf.keras.losses.BinaryCrossentropy()

        # the "real" labels to perform BCE on
        truth_real = tf.ones_like(disc_real_output)
        return bce(truth_real, disc_real_output)

    def upsample(inputs, skipinputs):
        """Returns output of upsampling chunk with addition of skip connection layer"""
        x = tf.image.resize(inputs, size=[5,7], method="nearest")
        #TODO: convolve skipinput to be correct size and then sum with x 
        #      not sure about filters, kernel size, strides, padding, or output padding here
        conv = Conv2DTranspose(filters=64, kernel_size=(1,1), padding="same", outpadding=1)
        y = conv(skipinputs)
        x += y
        return x
    
    def resnet(inputs, filter_size):
        """ Returns the output of a single resnet block """
        #TODO: Not sure if for each resnet block, the filter size decreases so I have it as a variable rn incase it halves.
        # RESNET18 does that but TBH the architecture is a little different from what I see in the paper.

        #NOTE: Not totally sure of the strides, this line is a bit confusing: 
        # "We halve feature map size in each layer except Layer-I using convolutions with stride of 2."
        # Re above: from the paper it looks like in layers 2, 3, and 4 the skip connecton in the first block must be convolved to the correct
        # size before concatenation

        #TODO: Ask about rescoping the model (pretrain some resnet layers, reduce learnable parameters)
        KERNEL_INIT = RandomNormal(stddev=0.02) 
        GAMMA_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        KERNEL_SIZE = 3
        
        mod_resnet = [
            Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding="same", 
                kernel_initializer=KERNEL_INIT, name="conv1"),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
            ReLU(), 
            MaxPool2d(strides=2),
            Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding="same", 
                kernel_initializer=KERNEL_INIT, name="conv2"),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
        ]
        
        output = inputs
        for layer in mod_resnet:
            output = layer(output)

        result = Concatenate()([output, inputs]) # "Skip concatenation" mentioned in paper, not sure if correctly implemented

        final_layer = [
            Conv2D(filters=filter_size, kernel_size=3, strides(1,1), padding="same", activation="relu", kernel_initializer=KERNEL_INIT)
        ]

        return result

        #TODO: NEED TO combine final_layer with above architecture


        # Vanilla RESNET18 Model from the paper here for reference.
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

Pix2Pix, could be relevant: https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
"""
class Discriminator(tf.keras.Model):
    def __init__(self, name=None):
        super(Discriminator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.name = name

        KERNEL_SIZE = 4
        STRIDE = 2
        RELU = .2

        # Weights initializer for the layers.
        KERNEL_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        # Gamma initializer for instance normalization.
        GAMMA_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        # Weird thing from the colab that I'll just leave here:
        #          img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
        
        self.layers = [
            # kernel_initializer 
            Conv2D(filters=64, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="block1_conv2"),
            LeakyReLU(RELU),

            Conv2D(filters=128, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="block2_conv1"),
            InstanceNormalization(gamma_initializer=GAMMA_INIT), # Alternative: InstanceNormalization
            LeakyReLU(RELU),

            Conv2D(filters=256, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="block2_conv1"),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
            LeakyReLU(RELU),

            Conv2D(filters=512, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block3_conv1"),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
            LeakyReLU(RELU),

            Conv2D(filters=1, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=1, padding="same", activation="sigmoid", name="block4_conv1")
        ]

        # Conv1:  in: 64, out:128
        # batchnorm 2d 128
        # leaky relu
        # ----
        # Conv2:  in: 128, out: 256
        # batchnorm 2d 256
        # leaky relu
        # ---
        # Conv3:  in: 256, out: 512
        # stride = 1
        # batch norm 2d: 512
        # leaky relu
        # --- 
        # Dense layer? Convolution from 512 to 1
        # 512->1, stride=1
        # sigmoid

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(disc_fake_output, disc_real_output):
        """ Loss function for model. """
        bce = tf.keras.losses.BinaryCrossentropy()

        # wants to maximize correctly classifying the reals as reals and the fakes as fakes
        # ground truth: all 1's for real, all 0's for fake
        # predicted: disc_real_output, disc_fake_output

        # the "real" labels to perform BCE on
        truth_real = tf.ones_like(disc_real_output)
        truth_fake = tf.zeros_like(disc_fake_output) 

        return bce(truth_real, disc_real_output) + bce(truth_fake, disc_fake_output)

        # Loss function for evaluating adversarial loss.
        # adv_loss_fn = keras.losses.MeanSquaredError()
        # def discriminator_loss_fn(real, fake):
        #     real_loss = adv_loss_fn(tf.ones_like(real), real)
        #     fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
        #     return (real_loss + fake_loss) * 0.5
