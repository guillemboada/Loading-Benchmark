""" 
This file contains the methods to create a TensorFlow dataset using .flow_from_directory()
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from IPython.display import clear_output
import configparser
config = configparser.ConfigParser()
config.read('parameters.ini')
HEIGHT = int(config['DEFAULT']['height'])
WIDTH = int(config['DEFAULT']['width'])


def get_generator(directory_images_path, directory_masks_path, batch_size):
    image_datagen = ImageDataGenerator(rescale=1/255)
    mask_datagen = ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(directory_images_path, class_mode=None, target_size=(HEIGHT, WIDTH), shuffle=False, batch_size=batch_size)
    mask_generator = mask_datagen.flow_from_directory(directory_masks_path, class_mode=None, target_size=(HEIGHT, WIDTH), shuffle=False, batch_size=batch_size)
    clear_output()

    train_generator = zip(image_generator, mask_generator)
    def generator():
        for image_batch, mask_batch in train_generator:
            # Reduce to a single channel
            mask_batch = mask_batch[:,:,:,0]
            # Map pixel labels from {1,2,3} to {0,1,2}
            mask_batch -= 1. 
            yield image_batch, mask_batch

    return generator

def get_dataset(directory_images_path, directory_masks_path, batch_size):
    dataset = tf.data.Dataset.from_generator(get_generator(directory_images_path, directory_masks_path, batch_size), output_types=(tf.float32, tf.float32))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    