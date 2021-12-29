import os
from glob import glob
import cv2
import random
import numpy as np
import tensorflow as tf
import configparser
config = configparser.ConfigParser()
config.read('parameters.ini')
HEIGHT = int(config['DEFAULT']['height'])
WIDTH = int(config['DEFAULT']['width'])


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

def load_samples(path_list, type):
    samples = []
    for path in path_list:
        if type=="images":
            sample = read_image(path)
        if type=="masks":
            sample = read_mask(path)

        sample = cv2.resize(sample, (WIDTH, HEIGHT))
        samples.append(sample)

    return samples

def generator_from_paths(x_path, y_path, batch_size):
    size = len(x_path)
    i = 0
    while True:
        real_batch_size = min(batch_size, size - i)
        x_data = load_samples(x_path[i:i+real_batch_size], "images")
        y_label = load_samples(y_path[i:i+real_batch_size], "masks")
        shape = x_data[0].shape
        batch_x = np.zeros((real_batch_size, shape[0], shape[1], shape[2])).astype('float32')
        batch_y = np.zeros((real_batch_size, shape[0], shape[1])).astype('float32')

        for k in range(i, i + real_batch_size):
            batch_x[k-i] = x_data[k-i]
            batch_y[k-i] = y_label[k-i]

        yield batch_x, batch_y
        i += real_batch_size
        if i >= size:
            i = 0
            # Shuffle data
            c = list(zip(x_path, y_path))
            random.shuffle(c)
            x_path, y_path = zip(*c)
            x_path = list(x_path)
            y_path = list(y_path)

def define_create_generator(path, batch_size):
    def create_generator():
        images_paths, masks_paths = get_data_paths(path)
        for image, mask in generator_from_paths(images_paths, masks_paths, batch_size):
            yield image, mask
    return create_generator

def get_data_paths(path):
    images_paths = glob(os.path.join(path, "images/*"))
    masks_paths = glob(os.path.join(path, "masks/*"))

    return images_paths, masks_paths

def get_generator(path, batch_size):
    create_generator = define_create_generator(path, batch_size)
    generator = create_generator()
    return generator

def get_dataset(path, batch_size):
    dataset = tf.data.Dataset.from_generator(define_create_generator(path, batch_size), output_types=(tf.float32, tf.float32))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
