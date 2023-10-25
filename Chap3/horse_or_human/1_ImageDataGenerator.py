import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds  # remove 'import resource' from tensorflow_datasets.core.shuffle


print(tf.__version__)


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#
# train_datagen = ImageDataGenerator(rescale=1/255)
#
# train_generator = train_datagen.flow_from_directory(
#     './datasets/training/',
#     target_size=(300, 300),
#     class_mode='binary'
# )
