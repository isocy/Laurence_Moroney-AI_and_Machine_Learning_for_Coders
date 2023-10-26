import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

import pathlib
import matplotlib.pyplot as plt


dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')

image_cnt = len(list(data_dir.glob('*/*.jpg')))
print(image_cnt)

roses = list(data_dir.glob('roses/*'))
img = PIL.Image.open(roses[0])
plt.imshow(img)
plt.show()

img = PIL.Image.open(roses[1])
plt.imshow(img)
plt.show()


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
