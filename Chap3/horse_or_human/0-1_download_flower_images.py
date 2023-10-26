import pathlib

from tensorflow.keras.utils import get_file


dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
file_path = get_file(origin=dataset_url, extract=True)

