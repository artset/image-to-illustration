import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import matplotlib as mpl

import hyperparameters as hp

from preprocess import Dataset
# from models import Ganilla, Generator
from ganilla import Ganilla, Generator, Discriminator

from skimage.transform import resize
from tensorboard_utils import  CustomModelSaver

from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError

#TODO: Change these before running evaluate!
DATASET_NAME = "elmer"
EPOCH = 10
SAVE_COUNT = 5

NUM_IMAGES = 100 # number of images to evaluate on, this can stay at 100

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


def train(model, photo_data, illo_data, checkpoint_path, logs_path, init_epoch, timestamp):
    print("Training model...")

    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath="checkpoints" + os.sep + timestamp + os.sep + "ganilla_{epoch:03d}", 
    #     save_freq='epoch', period=2)
    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        # checkpoint_callback
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    model.fit(
        tf.data.Dataset.zip((photo_data.train_data, illo_data.train_data)),
        epochs=hp.num_epochs,
        initial_epoch=init_epoch,
        callbacks=callback_list,
    )

def test(gen1, gen2, photo_data, illo_data, checkpoint):
    print("Testing model...")
    photo_data = photo_data.test_data
    illo_data = illo_data.test_data

    print("Generating images...")
    generate_illo(gen1, photo_data, if_save=True)

    print("Starting reconstruction evaluation...")
    reconstruction_eval(gen1, gen2, photo_data, if_save=True)

""" Converts to numy array to be an image"""
def convert_to_img(img_array):
    img_array = (img_array * 127.5 + 127.5).astype(np.uint8)
    return img_array

def generate_illo(generator, photo_data, if_save=False):
    NUM_IMG = 4
    count = 0
    for img in photo_data.take(SAVE_COUNT):
        count += 1

        prediction = generator(img, training=False)[0].numpy()
        prediction = convert_to_img(prediction)

        original = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8) # converts img to [0, 255]
        file_path = "generated" + os.sep + DATASET_NAME + os.sep + "epoch_" + str(EPOCH)

        if if_save:
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(file_path + os.sep + "{i}_generated.png".format(i=count))

            original = keras.preprocessing.image.array_to_img(original)
            original.save(file_path + os.sep + "{i}_original.png".format(i=count))

def reconstruction_eval(gen1, gen2, photo_data, if_save=False):
    count = 0

    ssim = 0
    mse = 0
    psnr = 0

    for img in photo_data.take(NUM_IMAGES):
        fake_illo = gen1(img, training=False)
        fake_photo = gen2(fake_illo, training=False)[0].numpy()

        fake_photo = convert_to_img(fake_photo)
        original = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8) # converts img to [0, 255]

        mse += mean_squared_error(original, fake_photo)
        ssim += structural_similarity(original, fake_photo, multichannel=True)
        psnr += peak_signal_noise_ratio(original, fake_photo)
        count += 1

        if if_save:
            file_path = "reconstructed" + os.sep + DATASET_NAME + os.sep + "epoch_" + str(EPOCH)
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            if count <= SAVE_COUNT:
                fake_photo = keras.preprocessing.image.array_to_img(fake_photo)
                fake_photo.save(file_path + os.sep + "{i}_cycled.png".format(i=count))

                original = keras.preprocessing.image.array_to_img(original)
                original.save(file_path + os.sep + "{i}_original.png".format(i=count))

        if count % 20 == 0:
            print(count, "images processed")
    mse = mse / (NUM_IMAGES)
    ssim = ssim / NUM_IMAGES

    print("MSE", mse)
    print("SSIM", ssim)
    print("PSNR", psnr)
    return mse


# Loss function for evaluating adversarial loss
# bce = keras.losses.MeanAbsoluteError()
bce = keras.losses.BinaryCrossentropy()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = bce(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = bce(tf.ones_like(real), real)
    fake_loss = bce(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss)  * .5 # Added this back



def main():
    """ Main function. """
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    checkpoint_path = "checkpoints" + os.sep + \
        "ganilla" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "ganilla" + \
        os.sep + timestamp + os.sep

    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    
    # Run script from location of run.py
    os.chdir(sys.path[0])

    ## # TEMP TESTIN
    # illo_data = Dataset("../data/train/illustration", "../data/test/illustration")
    # illo_data.check_images("../data/train/illustration/miyazaki")
    # illo_data.check_images("../data/test/illustration/miyazaki")
    # photo_data = Dataset("../data/train/landscape", "../data/test/landscape")
    # photo_data.check_images("../data/train/landscape/landscape")
    # photo_data.check_images("../data/test/landscape/landscape")
    # return

    print("Loading datasets...")
    illo_data = Dataset("../data/train/illustration", "../data/test/illustration")
    photo_data = Dataset("../data/train/landscape", "../data/test/landscape")

    np_photo = photo_data.train_data.as_numpy_iterator()
    print("number of photos", len(list((np_photo))))
    np_illo = illo_data.train_data.as_numpy_iterator()
    print("number of illustrations", len(list((np_illo))))

    gen_G = Generator("G")
    gen_F = Generator("F")
    disc_X = Discriminator("X")
    disc_Y = Discriminator("Y")

    gen_G(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    gen_F(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    disc_X(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    disc_Y(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

    gen_G.summary()
    disc_X.summary()

    if ARGS.load_checkpoint is not None:
        print("Loading checkpoint...")
        
        # checkpoints/ganilla/<TIMESTAMP>/epoch_2/g1_weights.h5
        # ARGS.load_checkpoint
        checkpoint = ARGS.load_checkpoint
        checkpoint_g1 = checkpoint + os.sep + "g1_weights.h5"
        checkpoint_g2 = checkpoint + os.sep + "g2_weights.h5"
        checkpoint_d1 = checkpoint + os.sep + "d1_weights.h5"
        checkpoint_d2 = checkpoint + os.sep + "d2_weights.h5"

        gen_G.load_weights(checkpoint_g1, by_name=False)
        gen_F.load_weights(checkpoint_g2, by_name=False)
        disc_X.load_weights(checkpoint_d1, by_name=False)
        disc_Y.load_weights(checkpoint_d2, by_name=False)

        absolute_path = os.path.abspath(ARGS.load_checkpoint)
        init_epoch = int(absolute_path[-2:])
        timestamp = os.path.basename(os.path.dirname(absolute_path))

    model = Ganilla(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )


    # Compile the model
    model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    if ARGS.evaluate:
        test(gen_G, gen_F, photo_data, illo_data, ARGS.load_checkpoint)
    else:
        train(model, photo_data, illo_data, checkpoint_path, logs_path, init_epoch, timestamp)
    
    ####### JUST TESTING GENERATOR
    # model = Generator()
    # model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    # checkpoint_path = "checkpoints" + os.sep + \
    #     "ganilla" + os.sep + timestamp + os.sep
    # logs_path = "logs" + os.sep + "ganilla" + \
    #     os.sep + timestamp + os.sep

# Make arguments global
ARGS = parse_args()

main()