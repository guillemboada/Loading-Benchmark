import os
from glob import glob
from time import time
import urllib.request
import tarfile
import shutil
import cv2
import numpy as np
import tensorflow as tf
import configparser
from subprocess import check_output
import psutil
import wandb


config = configparser.ConfigParser()
config.read('parameters.ini')
HEIGHT = int(config['DEFAULT']['height'])
WIDTH = int(config['DEFAULT']['width'])


def get_data_paths(path):
    images_paths = glob(os.path.join(path, "images/*"))
    masks_paths = glob(os.path.join(path, "masks/*"))

    return images_paths, masks_paths

def download_data(data_path):

    images_path = os.path.join(data_path, "images")
    masks_path = os.path.join(data_path, "masks")

    if os.path.isdir(images_path) and os.path.isdir(masks_path):
        print(f"Data is already in indicated path: ..\{os.path.basename(data_path)}")
    else:
        print("Downloading data...")
        start = time()
        # Download data
        IMAGES_LINK = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        MASKS_LINK = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        urllib.request.urlretrieve(IMAGES_LINK, os.path.join(data_path, IMAGES_LINK.split("/")[-1]))
        urllib.request.urlretrieve(MASKS_LINK, os.path.join(data_path, MASKS_LINK.split("/")[-1]))

        # Extract data
        for filename in os.listdir(data_path):
            if filename.endswith("tar.gz"):
                tar = tarfile.open(os.path.join(data_path, filename), "r:gz")
                tar.extractall(data_path)
                tar.close()

        # Structure data
        os.mkdir(masks_path)
        origin_masks_path = os.path.join(data_path, "annotations", "trimaps")
        for filename in os.listdir(origin_masks_path):
            if not filename.startswith("._"):
                os.rename(os.path.join(origin_masks_path, filename), os.path.join(masks_path, filename))

        # Clean up directory structure
        shutil.rmtree(os.path.join(data_path, "annotations"))
        os.remove(os.path.join(data_path, "annotations.tar.gz"))
        os.remove(os.path.join(data_path, "images.tar.gz"))
        for filename in os.listdir(images_path):
            if filename.endswith(".mat"):
                os.remove(os.path.join(images_path, filename))

        # Delete defective samples: cannot be loaded using cv2.imread()
        images_paths, masks_paths = get_data_paths(data_path)
        defective_samples = []
        for i,(image_path,mask_path) in enumerate(zip(images_paths, masks_paths)):
            try:
                image = read_image(image_path)
                mask = read_mask(mask_path)
            except:
                filename = os.path.basename(image_path)
                defective_sample = os.path.splitext(filename)[0]
                os.remove(os.path.join(images_path, defective_sample + ".jpg"))
                os.remove(os.path.join(masks_path, defective_sample + ".png"))

        print(f"- Done ({time() - start:.2f} seconds)")

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

def structure_data_in_subdirectories(path):
    images_path = os.path.join(path, "images")
    masks_path = os.path.join(path, "masks")
    directory_images_path = os.path.join(path, "directory_images")
    directory_masks_path = os.path.join(path, "directory_masks")
    subdirectory_images_path = os.path.join(path, "directory_images", "subdirectory_images")
    subdirectory_masks_path = os.path.join(path, "directory_masks", "subdirectory_masks")
    directories = [directory_images_path, directory_masks_path, subdirectory_images_path, subdirectory_masks_path]
    if os.path.isdir(subdirectory_images_path) and os.path.isdir(subdirectory_masks_path):
        print(f"Data is already in paths: ..\{os.path.basename(directory_images_path)}, ..\{os.path.basename(directory_masks_path)}")
    else:
        print("Structuring data in subdirectories...")
        start = time()
        for directory in directories:
            if not os.path.exists(directory):
                os.mkdir(directory)
        for filename in os.listdir(images_path):
            sample_name = os.path.splitext(filename)[0]
            shutil.copy(os.path.join(images_path, sample_name + ".jpg"), os.path.join(subdirectory_images_path, sample_name + ".jpg"))
            shutil.copy(os.path.join(masks_path, sample_name + ".png"), os.path.join(subdirectory_masks_path, sample_name + ".png"))
        print(f"- Done ({time() - start:.2f} seconds)")

class MonitorCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None): # on_train_batch_end
    gpu_memories = check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used"], encoding="utf-8")
    gpu_memory_usage_in_MiB = [int(s) for s in gpu_memories.split() if s.isdigit()][0]
    memory_usage = {"RAM (MB)": psutil.virtual_memory().used / (1024**2),
        "RAM (%)": psutil.virtual_memory().percent,
        "VRAM (MB) from tf": tf.config.experimental.get_memory_usage('GPU:0') / (1024**2),
        "VRAM (MB) from smi": gpu_memory_usage_in_MiB * 1.048576}
    wandb.log(memory_usage)