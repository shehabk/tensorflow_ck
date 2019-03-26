import tensorflow as tf
tf.enable_eager_execution()


import tensorflow.contrib.eager as tfe
import os
import numpy as np
from datasets.datagenerator import ImageDataGenerator

# Path to the textfiles for the trainings and validation set
root  = '/media/shehabk/D_DRIVE/codes/code_practice/tensorflow_ck/data/cropped_256'
train_file = 'image_lists/set1/train.txt'
val_file   = 'image_lists/set1/val.txt'


# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 7



def main():

    tr_data = ImageDataGenerator(train_file,
                                root = root,
                                mode='training',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                shuffle=True)

    # Training loop.
    for i, (xs, ys) in enumerate(tfe.Iterator(tr_data.data)):
        print (ys)
        break


if __name__ == "__main__":
    main()