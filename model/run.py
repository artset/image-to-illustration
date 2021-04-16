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

import hyperparameters as hp
from models import CycleGan, gen_F, gen_G, disc_X, disc_Y
from preprocess import Dataset
from skimage.transform import resize
from tensorboard_utils import  CustomModelSaver

from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np


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


# def train(model, datasets, checkpoint_path, logs_path, init_epoch):
#     """ Training routine. """

#     # Keras callbacks for training
#     callback_list = [
#         tf.keras.callbacks.TensorBoard(
#             log_dir=logs_path,
#             update_freq='batch',
#             profile_batch=0),
#         # ImageLabelingLogger(logs_path, datasets),
#         CustomModelSaver(checkpoint_path, hp.max_num_weights)
#     ]

#     # Include confusion logger in callbacks if flag set
#     if ARGS.confusion:
#         callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

#     train_data = np.zeros((5, 256, 256))
#     # Begin training
#     model.fit(
#         x = train_data,
#         # x=datasets.train_data,
#         # validation_data=datasets.test_data,
#         epochs=hp.num_epochs,
#         batch_size=None,
#         callbacks=callback_list,
#         initial_epoch=init_epoch,
#     )
#     model.summary()


# def test(model, test_data):
#     """ Testing routine. """

#     # Run model on test set
#     model.evaluate(
#         x=test_data,
#         verbose=1,
#     )
# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    # datasets = Datasets(ARGS.data)

    illo_data = Dataset("../data/train/illustration", "../data/test/illustration")
    photo_data = Dataset("../data/train/landscape", "../data/test/landscape")



    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn
    )


    cycle_gan_model.fit(
        tf.data.Dataset.zip((photo_data.train_data, illo_data.train_data)),
        epochs=1
    )

    # model = Ganilla()
    # # model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    # checkpoint_path = "checkpoints" + os.sep + \
    #     "ganilla" + os.sep + timestamp + os.sep
    # logs_path = "logs" + os.sep + "ganilla" + \
    #     os.sep + timestamp + os.sep

   
    # # Load checkpoints
    # if ARGS.load_checkpoint is not None:
    #     print("loading checkpoint...")
    #     model.load_weights(ARGS.load_checkpoint, by_name=False)
    #     # else:
    #     #     model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # # Make checkpoint directory if needed
    # if not ARGS.evaluate and not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)

    # print("compiling model graph...")
    # # Compile model graph
    # model.compile(
    #     # optimizer=model.optimizer,
    #     # loss=model.loss_fn,
    #     metrics=["gen_illos_loss", "gen_photos_loss", "disc_illos_loss", "disc_photos_loss"])

    # if ARGS.evaluate:
    #     test(model, datasets.test_data)

    #     # change the image path to be the image of your choice by changing
    #     # the lime-image flag when calling run.py to investigate
    #     # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg
    #     path = ARGS.data + os.sep + ARGS.lime_image
    #     LIME_explainer(model, path, datasets.preprocess_fn)
    # else:
    # train(model, datasets, checkpoint_path, logs_path, init_epoch)
    


# Make arguments global
ARGS = parse_args()

main()