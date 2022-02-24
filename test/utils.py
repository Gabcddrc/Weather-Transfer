from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import errno

def transform_all_images(data_path: str, width: int, height: int):
    """
    transforms all images in directory and subdirectories of data_path/images into resolution
    height x width. The resulting images are shown in the same folder structure with root data_path/images_height_width.
    """
    count = 0
    data_path = os.path.normpath(data_path)
    data_path = os.path.join(data_path, "images")
    new_data_path = os.path.join(os.path.split(data_path)[0], f"images_{width}_{height}")
    for root, dirs, files in tqdm(os.walk(data_path, topdown = False)):
        for name in tqdm(files, leave=False):
            image = Image.open(os.path.join(root, name))
            image = image.resize([width, height], Image.NEAREST)

            relative_path = os.path.relpath(root, data_path)
            file_loc = os.path.join(new_data_path, relative_path, name)
            try:
                os.makedirs(os.path.dirname(file_loc))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            image.save(file_loc)
            count += 1
            

if __name__ == "__main__":
    dataset_path = "/home/dominik/Documents/DeepLearning/bdd100k"
    transform_all_images(dataset_path, 320, 180)