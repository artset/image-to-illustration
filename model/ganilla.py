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
    ZeroPadding2D, Conv2DTranspose, UpSampling2D, Concatenate, LeakyReLU, ReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.losses import MeanAbsoluteError

import hyperparameters as hp





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

input_img_size = (256, 256, 3)
kernel_init = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)

def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


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

        # Weird thing from the colab that I'll just leave here:
        #          img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
        # ^ I believe this just instantiates the input as a tensor - KS

        self.architecture = [
            # kernel_initializer 
            Conv2D(filters=64, kernel_initializer=KERNEL_INIT, kernel_size=KERNEL_SIZE, strides=STRIDE, padding="same", name="poopyfartface"),
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
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(disc_real_output, disc_fake_output):
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


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = Discriminator(name="X")
disc_Y = Discriminator(name="Y")

# disc_X = get_discriminator(name="discriminator_X")
# disc_Y = get_discriminator(name="discriminator_Y")



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
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
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

    def train_step(self, batch_data):
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
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
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }