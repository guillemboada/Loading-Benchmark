""" 
This file contains the methods to create a TensorFlow dataset using .from_tensor_slices()
Reference: "TensorFlow Dataset Pipeline for Semantic Segmentation using tf.data API", by Idiot Developer (https://www.youtube.com/watch?v=C5CbsTDwQM0)
"""

import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
import configparser
config = configparser.ConfigParser()
config.read('parameters.ini')
HEIGHT = int(config['DEFAULT']['height'])
WIDTH = int(config['DEFAULT']['width'])


def get_data_paths(path):
    images_paths = glob(os.path.join(path, "images/*"))
    masks_directory_paths = glob(os.path.join(path, "masks/*"))

    return images_paths, masks_directory_paths

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)

    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x -= 1
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)

    return x

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        
        x = read_image(x)
        y = read_mask(y)

        return x, y
        
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([HEIGHT, WIDTH, 3])
    mask.set_shape([HEIGHT, WIDTH, 1])

    return image, mask

def define_dataset(x, y, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_dataset(path, batch_size, buffer_size):
    images_paths, masks_paths = get_data_paths(path)
    dataset = define_dataset(images_paths, masks_paths, batch_size, buffer_size)
    
    return dataset