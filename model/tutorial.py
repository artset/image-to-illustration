from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D, BatchNormalization, \
    ZeroPadding2D, Conv2DTranspose, UpSampling2D, Concatenate, LeakyReLU, ReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.losses import MeanAbsoluteError

import hyperparameters as hp

# Create the discriminator
# discriminator = keras.Sequential(
#     [
#         keras.Input(shape=(28, 28, 1)),
#         layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.GlobalMaxPooling2D(),
#         layers.Dense(1),
#     ],
#     name="discriminator",
# )

# # Create the generator
latent_dim = 128
# generator = keras.Sequential(
    # [
    #     keras.Input(shape=(latent_dim,)),
    #     # We want to generate 128 coefficients to reshape into a 7x7x128 map
    #     layers.Dense(7 * 7 * 128),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Reshape((7, 7, 128)),
    #     layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    # ],
#     name="generator",
# )



class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()

        self.discriminator = Discriminator()
        self.generator = Generator()
        self.latent_dim = latent_dim

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
        return self.lambda_cycle * self.cy_loss(real, cycled)

    @staticmethod
    def identity_loss(real, cycled):
        """
        Gives our model the property that generators will do this
        illustrator -> illustrator generator -> illustrator
        photo -> photo generator -> photo
        """
        return self.lambda_identity * self.lambda_cycle * self.id_loss(real, cycled)


    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real):
        print("real!!", len(real))
        photos = real[0]
        illos = real[1]
        print("illos shape", illos.shape)
        with tf.GradientTape(persistent=True) as tape:
            fake = self.generator(photos)
            disc_f_labels = self.discriminator(fake)
            disc_r_labels = self.discriminator(illos)
            print("fake", fake)
            print("dsc labels", disc_f_labels, disc_r_labels)

            d_loss = self.discriminator.loss_fn(disc_f_labels,disc_r_labels)
            g_loss = self.generator.loss_fn(disc_r_labels)
            print("d loss", d_loss, g_loss)
        print("before compute grads")
        grads_g1 = tape.gradient(g_loss, self.generator.trainable_variables)
        grads_d1 = tape.gradient(d_loss, self.discriminator.trainable_variables)
        print("after compute grads", grads_g1, grads_d1)

        # Apply gradients to generators and discriminators
        self.generator.optimizer.apply_gradients(zip(grads_g1, self.generator.trainable_variables))
        print("108")
        self.discriminator.optimizer.apply_gradients(zip(grads_d1, self.discriminator.trainable_variables))
        print("109")





"""
The Generator model, containing modified RESNET blocks.
"""
class Generator(tf.keras.Model):
    def __init__(self, name=None):
        super(Generator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        GAMMA_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        # self.block1 = [
        #     # Original model seems to have a reflection pad, but not sure why
        #     # Block 1
        #     Conv2D(filters=64, kernel_size=7, strides=1, padding="same", name="block1_conv1"),
        #     InstanceNormalization(gamma_initializer=GAMMA_INIT),
        #     ReLU() # seems to go after normalization not CONV2D, need to fact check if this is legit
        #     # MaxPool2D(2, name="block1_conv1"),   
        # ]

        # self.pre_upsample = [
        #     # Not sure about filters, strides, padding, or output padding here
        #     Conv2DTranspose(filters=512, kernel_size=(1,1), padding="same")
        # ]

        # self.pre_upsample_simple = [
        #     # Not sure about filters, strides, padding, or output padding here
        #     Conv2DTranspose(filters=256, kernel_size=(1,1), padding="same")
        # ]

        # self.post_upsample = {
        #     # Not sure about stride, padding, or output padding here
        #     Conv2DTranspose(filters=64, kernel_size=(1,1), strides=1, padding="same"),
        #     Conv2DTranspose(filters=64, kernel_size=(7,7), strides=1, padding="same")
        # }

    # simplified call func 
    def call_simple(self, x):
        """ Passes the image through the network. """
        # TODO: the TAN BLOCKS between skip connections appear to be conv layers needed to properly resize things so that they can be concatenated or summed togehter!!!!
        # Need to add this to the resnet structure overal ^^
        # for layer in self.block1:
        #     x = layer(x)

        # oen less resnet block
        #saving intermediate outputs for use in the upsampling skip connections
        # layer_1a_out = self.resnet(x, 64, "layer_1_a")
        # layer_1b_out = self.resnet(layer_1a_out, 64, "layer_b")
        # layer_2a_out = self.resnet(layer_1b_out, 64, "layer_2_a")
        # layer_2b_out = self.resnet(layer_2a_out, 128, "layer_b")
        # layer_3a_out = self.resnet(layer_2b_out, 128, "layer_2_a")

        # #  layer 3b
        # x = self.resnet(layer_3a_out, 256, "layer_b")

        # for layer in self.pre_upsample_simple:
        #     x = layer(x)

        # # Original code seems to upsample twice 
        # x = self.upsample(x, layer_2b_out, 256)
        # x = self.upsample(x, layer_1b_out, 128)
        # # Not sure about the size
        # up = UpSampling2D(size=(2,2), interpolation="nearest")
        # x = up(x)
        
        # for layer in self.post_upsample(x):
        # #     x = layer(x)

        
        return x

    def call(self, x):
        """ Passes the image through the network. """
        # TODO: the TAN BLOCKS between skip connections appear to be conv layers needed to properly resize things so that they can be concatenated or summed togehter!!!!
        # Need to add this to the resnet structure overal ^^
        # for layer in self.block1:
        #     x = layer(x)

        # #TODO: a guess for the 4 downsampling blocks: call resnet on on 64, 128, 256 , 512
        # # Original code seems do downsample twice -- currently upsampling and downsampling four times to match the diagram on page 5
        # # Saving intermediate outputs for use in the upsampling skip connections
        # layer_1a_out = self.resnet(x, 64, "layer_1_a")
        # layer_1b_out = self.resnet(layer_1a_out, 64, "layer_b")
        # layer_2a_out = self.resnet(layer_1a_out, 64, "layer_2_a")
        # layer_2b_out = self.resnet(layer_2a_out, 128, "layer_b")
        # layer_3a_out = self.resnet(layer_2b_out, 128, "layer_2_a")
        # layer_3b_out = self.resnet(layer_3a_out, 256, "layer_b")
        # layer_4a_out = self.resnet(layer_3b_out, 256, "layer_2_a")
        # x = self.resnet(layer_2a_out, 512, "layer_b")

        # for layer in self.pre_upsample:
        #     x = layer(x)

        # # Original code seems to upsample twice 
        # x = self.upsample(x, layer_3b_out, 512)
        # x = self.upsample(x, layer_2b_out, 256)
        # x = self.upsample(x, layer_1b_out, 128)
        # # Not sure about the size
        # up = UpSampling2D(size=(2,2), interpolation="nearest")
        # x = up(x)
        
        # for layer in self.post_upsample(x):
        #     x = layer(x)


        # temp =     [
        # # keras.Input(shape=(latent_dim,)),
        # # We want to generate 128 coefficients to reshape into a 7x7x128 map
        # layers.Dense(7 * 7 * 128),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Reshape((7, 7, 128)),
        # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        # ]
        # print(temp)
        # for layer in temp:
        #     x = layer(x)
        #     print("x shape", x.shape)
        return x

    @staticmethod
    def loss_fn(disc_real_output):
        """ Loss function for model. """
        bce = tf.keras.losses.BinaryCrossentropy()

        # the "real" labels to perform BCE on
        truth_real = tf.ones_like(disc_real_output)
        return bce(truth_real, disc_real_output)

    def upsample(inputs, skipinputs, filter_size):
        """Returns output of upsampling chunk with addition of skip connection layer"""
        up = UpSampling2D(size=(2,2), interpolation="nearest") #Might need to change the data_format param based on how our data is structured
        #TODO: convolve skipinput to be correct size and then sum with x 
        #paper seems to say kernel_size = 1,1 but repo says 3,3
        conv = Conv2DTranspose(filters=filter_size, kernel_size=(1,1), stride=1, padding="same", outpadding=1)
        x = up(inputs)
        y = conv(skipinputs)
        x += y
        return x
    
    def resnet(inputs, filter_size, lay_type):
        """ Returns the output of a single resnet block """
        #TODO: Not sure if for each resnet block, the filter size decreases so I have it as a variable rn incase it halves.
        # RESNET18 does that but TBH the architecture is a little different from what I see in the paper.

        #NOTE: Not totally sure of the strides, this line is a bit confusing: 
        # "We halve feature map size in each layer except Layer-I using convolutions with stride of 2."
        # Re above: from the paper it looks like in layers 2, 3, and 4 the skip connecton in the first block must be convolved to the correct
        # size before concatenation

        #TODO: Ask about rescoping the model (pretrain some resnet layers, reduce learnable parameters)
        KERNEL_INIT = tf.keras.initializers.RandomNormal(stddev=0.02) 
        GAMMA_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        KERNEL_SIZE = 3
        
        if lay_type == "layer_1_a" or type == "layer_b": 
            mod_resnet = [
                Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(1,1), padding="same", 
                    kernel_initializer=KERNEL_INIT, name="conv1"),
                InstanceNormalization(gamma_initializer=GAMMA_INIT),
                ReLU(), 
                MaxPool2d(strides=2),
                Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(1,1), padding="same", 
                    kernel_initializer=KERNEL_INIT, name="conv2"),
                InstanceNormalization(gamma_initializer=GAMMA_INIT),
            ]
        elif lay_type == "layer_2_a":
            mod_resnet = [
                Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(2,2), padding="same", 
                    kernel_initializer=KERNEL_INIT, name="conv1"),
                InstanceNormalization(gamma_initializer=GAMMA_INIT),
                ReLU(), 
                MaxPool2d(strides=2),
                Conv2D(filters=filter_size, kernel_size=KERNEL_SIZE, strides=(1,1), padding="same", 
                    kernel_initializer=KERNEL_INIT, name="conv2"),
                InstanceNormalization(gamma_initializer=GAMMA_INIT),
            ]
        
        output = inputs
        for layer in mod_resnet:
            output = layer(output)

        size_mod = Conv2D(filters=filter_size, kernel_size=3, stride=(1,1), padding="same", activation="relu", kernel_initializer=KERNEL_INIT)
        
        if lay_type == "layer_2_a": 
            inputs = size_mod(inputs)

        result = Concatenate()([output, inputs]) # "Skip concatenation" mentioned in paper, not sure if correctly implemented

        final_layer = [
            Conv2D(filters=filter_size, kernel_size=3, stride=(1,1), padding="same", activation="relu", kernel_initializer=KERNEL_INIT)
        ]
        result = final_layer[0](result)

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
        # for layer in self.architecture:
        #     x = layer(x)
        x = Dense(100)(x)
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
