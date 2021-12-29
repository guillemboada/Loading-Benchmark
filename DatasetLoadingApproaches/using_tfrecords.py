import os
from glob import glob
import tensorflow as tf
import configparser
config = configparser.ConfigParser()
config.read('parameters.ini')
HEIGHT = int(config['DEFAULT']['height'])
WIDTH = int(config['DEFAULT']['width'])


def parse_tfr_element_for_segmentation(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width':tf.io.FixedLenFeature([], tf.int64),
      'depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'raw_mask':tf.io.FixedLenFeature([], tf.string)
    }
    
  content = tf.io.parse_single_example(element, data)
  
  height = content['height']
  width = content['width']
  depth = content['depth']
  raw_mask = content['raw_mask']
  raw_image = content['raw_image']
  
  #get our 'feature'-- our image -- and reshape it appropriately
  image = tf.io.parse_tensor(raw_image, out_type=tf.float32)
  image = tf.reshape(image, shape=[height, width, depth])

  mask = tf.io.parse_tensor(raw_mask, out_type=tf.float32)
  mask = tf.reshape(mask, shape=[height, width, 1])

  return (image, mask)

def get_dataset(tfrecords_path, pattern, batch_size, buffer_size):
  files = glob(os.path.join(tfrecords_path, f"{pattern}"), recursive=False)

  #create the dataset
  dataset = tf.data.TFRecordDataset(files)

  #pass every single feature through our mapping function
  dataset = dataset.map(
      parse_tfr_element_for_segmentation
  )

  dataset = (dataset
    .shuffle(buffer_size)
    .batch(batch_size)
    .repeat()
    .prefetch(tf.data.AUTOTUNE))
    
  return dataset
