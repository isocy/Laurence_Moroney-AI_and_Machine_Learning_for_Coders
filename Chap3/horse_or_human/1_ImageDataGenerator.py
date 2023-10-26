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

# roses = list(data_dir.glob('roses/*'))
# img = PIL.Image.open(roses[0])
# plt.imshow(img)
# plt.show()
#
# img = PIL.Image.open(roses[1])
# plt.imshow(img)
# plt.show()


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

for image_batches, label_batches in train_ds:
    print(image_batches.shape)
    print(label_batches.shape)
    break


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
