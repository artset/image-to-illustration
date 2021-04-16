"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import tensorflow_addons as tfa
 
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D, BatchNormalization, \
    ZeroPadding2D, Conv2DTranspose, UpSampling2D, Concatenate, LeakyReLU, ReLU, Activation
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.losses import MeanAbsoluteError

import hyperparameters as hp


# Need this for now -- do not change
class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


class Generator(tf.keras.Model):
    def __init__(self, name=None):
        super(Generator, self).__init__()
        self.gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.downsampling = [
            ReflectionPadding2D(padding=(3, 3)),
            layers.Conv2D(64, (7, 7), kernel_initializer=self.kernel_init, use_bias=False),
            tfa.layers.InstanceNormalization(gamma_initializer=self.gamma_init),
            layers.Activation("relu"),
            # downsampling blocks
            Conv2D(128, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
            ReLU(),

            Conv2D(256, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
            ReLU(),
        ]

        self.resnet1 = [
            ReflectionPadding2D(),
            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="valid", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
            ReLU(),

            ReflectionPadding2D(),
            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="valid", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
        ]

        self.resnet2 = [
            ReflectionPadding2D(),
            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="valid", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
            ReLU(),

            ReflectionPadding2D(),
            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="valid", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
        ]

        self.upsampling = [
            Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
            ReLU(),

            Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer=self.kernel_init, use_bias=False),
            InstanceNormalization(gamma_initializer=self.gamma_init),
            ReLU(),

            ReflectionPadding2D(padding=(3, 3)),
            layers.Conv2D(3, (7, 7), padding="valid"),
            layers.Activation("tanh")
        ]

    def call(self, x):
        num_downsampling_blocks = 2
        num_residual_blocks = 9
        num_upsample_blocks = 2
        print("----downsampling---")
        for l in self.downsampling:
            print("x", x.shape)
            x = l(x)


        original = tf.identity(x)
        print("---- resnet1 ---")
        for l in self.resnet1:
            print("x", x.shape)
            x = l(x)
        x = layers.add([original, x])
        original = tf.identity(x)

        print("---- resnet2 ---")
        for l in self.resnet2:
            print("x", x.shape)
            x = l(x)
        x = layers.add([original, x])

        print("------ upsampling -- ")
        # Final block
        for l in self.upsampling:
            print("x", x.shape)
            x = l(x)

        return x

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
        
        KERNEL_SIZE = 4
        STRIDE = 2
        RELU = .2

        # Weights initializer for the layers.
        KERNEL_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        # Gamma initializer for instance normalization.
        GAMMA_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        self.architecture = [
            # kernel_initializer 
            Conv2D(filters=64, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="block1_conv1"),
            LeakyReLU(RELU),

            Conv2D(filters=128, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="block2_conv1", use_bias=False),
            InstanceNormalization(gamma_initializer=GAMMA_INIT), # Alternative: InstanceNormalization
            LeakyReLU(RELU),

            Conv2D(filters=256, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="block2_conv1", use_bias=False),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
            LeakyReLU(RELU),

            Conv2D(filters=512, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=1, padding="same", name="block3_conv1", use_bias=False),
            InstanceNormalization(gamma_initializer=GAMMA_INIT),
            LeakyReLU(RELU),

            Conv2D(filters=1, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=1, padding="same", activation="sigmoid", name="block4_conv1")
        ]

    def call(self, x):
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(disc_real_output, disc_fake_output):
        """ Loss function for model. """
        bce = tf.keras.losses.BinaryCrossentropy()
        # the "real" labels to perform BCE on
        truth_real = tf.ones_like(disc_real_output)
        truth_fake = tf.zeros_like(disc_fake_output) 

        return bce(truth_real, disc_real_output) + bce(truth_fake, disc_fake_output)

gen_G = Generator("G")
gen_F = Generator("F")

# Get the discriminators
disc_X = Discriminator(name="X")
disc_Y = Discriminator(name="Y")

class Ganilla(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(Ganilla, self).__init__()
        self.g1 = generator_G
        self.g2 = generator_F
        self.d1 = discriminator_X
        self.d2 = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(Ganilla, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, data):
        real_photo, real_illo = data

        with tf.GradientTape(persistent=True) as tape:
            # Generate images
            fake_y = self.g1(real_photo, training=True)
            fake_x = self.g2(real_illo, training=True)

            # Discriminator output
            disc_real_x = self.d1(real_photo, training=True)
            disc_fake_x = self.d1(fake_x, training=True)

            disc_real_y = self.d2(real_illo, training=True)
            disc_fake_y = self.d2(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycled_x = self.g2(fake_y, training=True)
            cycled_y = self.g1(fake_x, training=True)
            cycle_loss_G = self.cycle_loss_fn(real_illo, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_photo, cycled_x) * self.lambda_cycle

            # Generator identity loss
            same_x = self.g2(real_photo, training=True)
            same_y = self.g1(real_illo, training=True)
            id_loss_G = (
                self.identity_loss_fn(real_illo, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_photo, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.g1.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.g2.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.d1.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.d2.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.g1.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.g2.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.d1.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.d2.trainable_variables)
        )

        return {
            "gen1_loss": total_loss_G,
            "gen2_loss": total_loss_F,
            "disc1_loss": disc_X_loss,
            "disc2_loss": disc_Y_loss,
        }