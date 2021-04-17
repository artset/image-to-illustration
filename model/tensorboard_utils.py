"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import io
import os
import re
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp


def plot_to_image(figure):
    """ Converts a pyplot figure to an image tensor. """

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


class ImageLabelingLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a plot of test images and their
    predicted labels for viewing in Tensorboard. """

    def __init__(self, logs_path, datasets):
        super(ImageLabelingLogger, self).__init__()

        self.datasets = datasets
        self.task = datasets.task
        self.logs_path = logs_path

        print("Done setting up image labeling logger.")

    def on_epoch_end(self, epoch, logs=None):
        self.log_image_labels(epoch, logs)

    def log_image_labels(self, epoch_num, logs):
        """ Writes a plot of test images and their predicted labels
        to disk. """

        fig = plt.figure(figsize=(9, 9))
        count = 0
        for batch in self.datasets.train_data:
            for i, image in enumerate(batch[0]):
                plt.subplot(5, 5, count+1)

                correct_class_idx = batch[1][i]
                probabilities = self.model(np.array([image])).numpy()[0]
                predict_class_idx = np.argmax(probabilities)

                if self.task == '1':
                    image = np.clip(image, 0., 1.)
                    plt.imshow(image, cmap='gray')
                else:
                    # Undo VGG preprocessing
                    mean = [103.939, 116.779, 123.68]
                    image[..., 0] += mean[0]
                    image[..., 1] += mean[1]
                    image[..., 2] += mean[2]
                    image = image[:, :, ::-1]
                    image = image / 255.
                    image = np.clip(image, 0., 1.)

                    plt.imshow(image)

                is_correct = correct_class_idx == predict_class_idx

                title_color = 'g' if is_correct else 'r'

                plt.title(
                    self.datasets.idx_to_class[predict_class_idx],
                    color=title_color)
                plt.axis('off')

                count += 1
                if count == 25:
                    break

            if count == 25:
                break

        figure_img = plot_to_image(fig)

        file_writer_il = tf.summary.create_file_writer(
            self.logs_path + os.sep + "image_labels")

        with file_writer_il.as_default():
            tf.summary.image("Image Label Predictions",
                             figure_img, step=epoch_num)



class CustomModelSaver(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, task, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.task = task
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """ At epoch end, weights are saved to checkpoint directory. """

        print("on epoch end", epoch)
        epoch_str = epoch + 1

        
        if (epoch_str % 10 == 0):
            # make directory named timestamp/epoch_{epoch_str}
            if (epoch_str < 10):
                new_dir = "epoch_" + "0" + str(epoch_str)
            else:
                new_dir = "epoch_" + str(epoch_str)

            os.mkdir(self.checkpoint_dir + os.sep + new_dir)


            # In that directory, save it
            save_name = "weights.h5"

            self.model.g1.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "g1_" + save_name)
            self.model.g2.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "g2_" + save_name)
            self.model.d1.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "d1_" + save_name)
            self.model.d2.save_weights(self.checkpoint_dir + os.sep + new_dir + os.sep + "d2_" + save_name)


        # Only save weights if test accuracy exceeds the previous best
        # # weight file
        # if cur_acc > max_acc:
        #     save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(
        #         epoch, cur_acc)

        #     if self.task == '1':
        #         self.model.save_weights(
        #             self.checkpoint_dir + os.sep + "your." + save_name)
        #     else:
        #         # Only save weights of classification head of VGGModel
        #         self.model.head.save_weights(
        #             self.checkpoint_dir + os.sep + "vgg." + save_name)

        #     # Ensure max_num_weights is not exceeded by removing
        #     # minimum weight
        #     if self.max_num_weights > 0 and \
        #             num_weights + 1 > self.max_num_weights:
        #         os.remove(self.checkpoint_dir + os.sep + min_acc_file)