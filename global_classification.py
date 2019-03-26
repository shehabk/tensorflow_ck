import matplotlib.pyplot as plt
import tensorflow as tf
from datasets.dataloader import DataLoader
import numpy as np
import os

plt.ioff()

IMAGE_DIR_PATH = 'data/training/images'
MASK_DIR_PATH = 'data/training/masks'

pascal_palette = np.array([
        [  0,   0,   0],
        [128,   0,   0],
        [  0, 128,   0],
        [128, 128,   0],
        [  0,   0, 128],
        [128,   0, 128],
        [  0, 128, 128],
        [128, 128, 128],
        [ 64,   0,   0],
        [192,   0,   0],
        [ 64, 128,   0],
        [192, 128,   0],
        [ 64,   0, 128],
        [192,   0, 128],
        [ 64, 128, 128],
        [192, 128, 128],
        [  0,  64,   0],
        [128,  64,   0],
        [  0, 192,   0],
        [128, 192,   0],
        [  0,  64, 128]], dtype=np.uint8)

# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths  = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

# Where image_paths[0] = 'data/training/images/image_0.png' 
# And mask_paths[0] = 'data/training/masks/mask_0.png'

# Initialize the dataloader object
dataset = DataLoader(image_paths=image_paths,
                     mask_paths=mask_paths,
                     image_size=[256, 256],
                     crop_percent=0.8,
                     channels=[3, 3],
                     palette=pascal_palette,
                     seed=47)

# Parse the images and masks, and return the data in batches, augmented optionally.
data, init_op = dataset.data_batch(augment=True, 
                                   shuffle=True,
                                   one_hot_encode=True,
                                   batch_size=BATCH_SIZE,
                                   num_threads=4,
                                   buffer=60)


with tf.Session() as sess:
  # Initialize the data queue
  sess.run(init_op)
  # Evaluate the tensors
  aug_image, aug_mask = sess.run(data)
                                 
  # Do whatever you want now, like creating a feed dict and train your models,
  # You can also directly feed in the tf tensors to your models to avoid using a feed dict.