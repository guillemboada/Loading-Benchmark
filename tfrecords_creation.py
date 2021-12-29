import os
from glob import glob
from time import time
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import configparser
config = configparser.ConfigParser()
config.read('parameters.ini')
HEIGHT = int(config['DEFAULT']['height'])
WIDTH = int(config['DEFAULT']['width'])


def get_data_paths(path):
    images_paths = glob(os.path.join(path, "images/*"))
    masks_paths = glob(os.path.join(path, "masks/*"))

    return images_paths, masks_paths

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = x.astype(np.float32) / 255.

    return x
    
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (WIDTH, HEIGHT))
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32) - 1

    return x

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_image_for_segmentation(image, mask):
  
  #define the dictionary -- the structure -- of our single example
  data = {
        'height' : _int64_feature(image.shape[0]),
        'width' : _int64_feature(image.shape[1]),
        'depth' : _int64_feature(image.shape[2]),
        'raw_image' : _bytes_feature(serialize_array(image)),
        'raw_mask' : _bytes_feature(serialize_array(mask))
    }
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out

def dynamic_write_images_to_tfr(path, max_files, tfrecords_path, filename):

  images_paths, masks_paths = get_data_paths(path)

  #determine the number of shards (single TFRecord files) we need:
  N = len(images_paths)
  splits = (N//max_files) + 1 #determine how many tfr shards are needed
  if N%max_files == 0:
    splits-=1
    
  unix_time = str(time()).split(".")[0]
  file_count = 0
  for i in tqdm(range(max_files)):
    current_shard_name = os.path.join(tfrecords_path, f"{filename}_{i+1}_{max_files}_{unix_time}.tfrecords")
    writer = tf.io.TFRecordWriter(current_shard_name)

    current_shard_count = 0
    while current_shard_count < splits: #as long as our shard is not full
      #get the index of the file that we want to parse now
      index = i*splits+current_shard_count
      if index == N: #when we have consumed the whole data, preempt generation
        break
      current_image = read_image(images_paths[index])
      current_mask =  read_mask(masks_paths[index])

      #create the required Example representation
      out = parse_single_image_for_segmentation(image=current_image, mask=current_mask)

      writer.write(out.SerializeToString())
      current_shard_count += 1
    file_count += 1

    writer.close()
  print(f"\nWrote {N} elements to {max_files} TFRECORDS files. There are up to {splits} datapoints / file.")
