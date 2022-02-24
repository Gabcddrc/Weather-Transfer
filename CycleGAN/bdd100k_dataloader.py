import torch
from PIL import Image
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union
import os
from scalabel.label.io import load
from os import listdir
from os.path import isfile, join

weather_label_to_id = {
    "clear": 0,
    "rainy": 1,
    "undefined": 2,
    "snowy": 3,
    "overcast": 4,
    "partly cloudy": 5,
    "foggy": 6
}


semantic_labels = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    ("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    ("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    ("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    ("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    ("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    ("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    ("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    ("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    ("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    ("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    ("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    ("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    ("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    (
        "guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)
    ),
    ("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    ("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    ("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    ("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    ("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    ("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    ("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    ("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    ("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    ("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    (
        "traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)
    ),
    ("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    ("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    (
        "traffic sign frame",
        27,
        255,
        "object",
        3,
        False,
        True,
        (250, 170, 250),
    ),
    ("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    ("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    ("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    ("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    ("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    ("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    ("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    ("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    ("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    ("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    ("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    ("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    ("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]


class Bdd100kDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs: List[str], image_dataset_path: str, labels_path: str, keep_in_memory: bool = False):
        """
        Creates a Bdd100k dataset. One dataset should be created for train, test, validation.

        Keyword arguments:
            list_IDs: List[str] -- a list of strings associated with the dataset. The dataset will read all files which are specified by the list_IDs.
            image_dataset_path: str -- the path to the dataset
            labels_path: str -- the path to the labels .json
            keep_in_memory -- whether the whole dataset should be read into the memory or read from disk whenever __getitem__ is called.
        """
        self.image_dataset_path = image_dataset_path

        # this function takes up to 18GB of memory temporarily
        dataset = load(labels_path, 2)
#         print(dataset.frames[0])
        self.weather_labels = {}
        self.length = 0
        
        # all files in the images_dataset_path
        all_available_images = [f for f in listdir(image_dataset_path) if isfile(join(image_dataset_path, f))]
        available_ids = []
        def add_to_labels(frame):
            if (frame.name[:-4] in list_IDs and frame.name in all_available_images):
                self.weather_labels[frame.name[:-4]] = weather_label_to_id[frame.attributes["weather"]]
                self.length += 1
                available_ids.append(frame.name[:-4])                
        
        
        # go through all of datasets ids and check for their existence in the image_dataset_path
        # if they are in the path and in our list_IDs then we can add the image to self
        # [add_to_labels(frame) for frame in dataset.frames]
        list(map(add_to_labels, dataset.frames))
        # print(len(dataset.frames), len(all_available_images), len(available_ids))
        
        self.list_IDs = available_ids
        if self.length != len(self.list_IDs):
            print(f"The length of the dataset ({self.length}) does not match the number of IDs ({len(self.list_IDs)}) provided. Continueing anyways")
        
        self.keep_in_memory = keep_in_memory
        if(keep_in_memory):
            self.images = []
            # load all images into
            for id in self.list_IDs:
                rgb_image = torch.tensor(self.load_image(id))

                # include more data in this tupel if needed
                datum = (rgb_image)
                self.data.append(datum)
                
        
           
    def load_image(self, image_name: str) -> np.ndarray:
        return self.load_rgb_image_file(os.path.join(self.image_dataset_path, image_name+".jpg"))

    def load_semantic_map(self, image_name: str, data_set: List[str]) -> np.ndarray:
        return self.load_image_file(os.path.join(semantic_maps_paths[data_set], image_name+".png"))

    def load_rgb_image_file(self, path: Union[str]) -> np.ndarray:
        im_frame = Image.open(path).convert("RGB")
        np_frame = np.array(im_frame)
        return np_frame
    
    def semantic_rgb_to_grayscale_map(self, image):
        semantic_map = torch.zeros([image.shape[0], image.shape[1]], dtype=torch.float32)
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                semantic_map[y, x] = self.rgb_to_id(image[y,x])
        return semantic_map
    
    def rgb_to_id(self, rgb_pixel):
        for label in semantic_labels:
            if rgb_pixel == label[7]:
                return label[1]
                

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        """Generates one sample of data
        a sample is a tuple (X, y)
        - X a torch.tensor with shape (height, width, rgb_channels + semantic_channel = 4)
        - y the id of the weather
        """
        if (self.keep_in_memory):
            image_rgb = self.images[index][0]
            #TODO fill with semantic image
            image_semantic = torch.zeros(image_rgb.shape[0],image_rgb.shape[1],1)
        else:
            # print(self.load_image(self.list_IDs[index]).shape)
            image_rgb = torch.from_numpy(self.load_image(self.list_IDs[index]))
            # load semantic image from the disk
            image_semantic = torch.zeros(image_rgb.shape[0],image_rgb.shape[1],1)
        
        y = self.weather_labels[self.list_IDs[index]]
        return torch.cat([image_rgb, image_semantic], dim=2), y