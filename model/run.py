"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

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
from ganilla import Ganilla, gen_G, gen_F, disc_X, disc_Y

from skimage.transform import resize
from tensorboard_utils import  CustomModelSaver

from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError



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


def train(model, photo_data, illo_data, checkpoint_path, logs_path, init_epoch):

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    model.fit(
        tf.data.Dataset.zip((photo_data.train_data, illo_data.train_data)),
        epochs=hp.num_epochs,
        initial_epoch=init_epoch,
        callbacks=callback_list,
    )

def test(model, photo_data, illo_data, checkpoint):
    print("hello)")
    """ Testing routine. """

    # Run model on test set
    # model.evaluate(
    #     x=test_data,
    #     verbose=1,
    # )


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
    return (real_loss + fake_loss) # * .5? 



def main():
    """ Main function. """
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)
        init_epoch = int(ARGS.load_checkpoint[-2:])
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

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

    illo_data = Dataset("../data/train/illustration", "../data/test/illustration")
    photo_data = Dataset("../data/train/landscape", "../data/test/landscape")

    model = Ganilla(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    if ARGS.load_checkpoint is not None:
        print("loading checkpoint...")
        
        # checkpoints/ganilla/<TIMESTAMP>/epoch_2/g1_weights.h5
        # ARGS.load_checkpoint
        checkpoint = ARGS.load_checkpoint
        checkpoint_g1 = checkpoint + "g1_weights.h5"
        checkpoint_g2 = checkpoint + "g2_weights.h5"
        checkpoint_d1 = checkpoint + "d1_weights.h5"
        checkpoint_d2 = checkpoint + "d2_weights.h5"

        model.g1.load_weights(checkpoint_g1, by_name=False)
        model.g2.load_weights(checkpoint_g2, by_name=False)
        model.d1.load_weights(checkpoint_d1, by_name=False)
        model.d2.load_weights(checkpoint_d2, by_name=False)

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
        test(model, photo_data, illo_data, ARGS.load_checkpoint)
    else:
        train(model, photo_data, illo_data, checkpoint_path, logs_path, init_epoch)
    
    ######## JUST TESTING GENERATOR
    # model = Generator()
    # model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    # checkpoint_path = "checkpoints" + os.sep + \
    #     "ganilla" + os.sep + timestamp + os.sep
    # logs_path = "logs" + os.sep + "ganilla" + \
    #     os.sep + timestamp + os.sep

# Make arguments global
ARGS = parse_args()

main()