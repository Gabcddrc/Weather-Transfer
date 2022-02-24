import random

import torchvision.transforms
from PIL import Image
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union
import os
from scalabel.label.io import load
from os import listdir
from os.path import isfile, join
import torch
from utils.config import Config
from torchvision.transforms import transforms as T
import torchvision.transforms.functional as F
from torch.nn.functional import one_hot

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


def load_list_image_names(path: str) -> List[str]:
    """
    reads predefined set of image names for train/val/test and returns list of it
    """
    with open(path) as f:
        lines = f.read().splitlines()
        return lines


def adjust_brightness_random(img):
    return F.adjust_brightness(img, brightness_factor=random.uniform(0.5, 1.5))


def adjust_contrast_random(img):
    return F.adjust_contrast(img, contrast_factor=random.uniform(0.5, 1.5))


def adjust_hue_random(img):
    return F.adjust_hue(img, hue_factor=random.uniform(-0.25, 0.25))


def adjust_saturation_random(img):
    return F.adjust_saturation(img, saturation_factor=random.uniform(0.5, 1.5))


def adjust_sharpness_random(img):
    return F.adjust_sharpness(img, sharpness_factor=random.uniform(0.5, 1.5))


class Bdd100kDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, config, mode: str, weather: str, keep_in_memory: bool = False):
        """
        Creates a Bdd100k dataset. The same dataset class is used by generator and discriminator.
        Inputs:
        - config: is the instantiation of the Config() class
        - mode: is one of ('train', 'test', 'val')
        - weather: is one of the weather classes supported by the dataset.
        It is one of ('clear', 'rainy', 'undefined', 'snowy', 'overcast', 'partly cloudy', 'foggy')
        - keep_in_memory: is a boolean. True if we want the images to get loaded in memory all together in the first
        place. False if each image is loaded in memory when called by the dataloader.
        """
        self.config = config
        self.pat = config.probability_augmentation_transform
        self.keep_in_memory = keep_in_memory
        self.weather = weather
        self.weather_id = weather_label_to_id[weather]
        # transforms for data augmentation
        t1 = T.RandomHorizontalFlip(0.5)
        t2 = T.Pad(3)
        t3 = T.RandomCrop((224, 400))

        self.horizontal_flip = T.RandomHorizontalFlip(0.5)
        resize_factor = int(224 * 1.12)
        self.resize_img = T.Resize(resize_factor, torchvision.transforms.InterpolationMode.BICUBIC)

        self.resize_map = T.Resize(resize_factor, torchvision.transforms.InterpolationMode.NEAREST)
        self.random_crop = T.RandomCrop((224, 400))
        # self.transforms = torch.nn.Sequential(
        #     T.Resize(int(224 * 1.12), Image.BICUBIC),
        #     T.RandomCrop((224, 400)),
        #     t1,
        #     # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # )
        # Get information from config
        if mode == 'train':
            self.image_path = config.dataset_train
            # TODO: add multi-weather
            self.segmentation_classid_paths = config.segmentation_classid_train + weather
            # self.segmentation_classid_paths = []
            # self.segmentation_colormap_paths = []
            # for el in weather:
            #     self.segmentation_classid_path.append(config.segmentation_classid_train + el)
            #     self.segmentation_colormap_path.append(config.segmentation_colormaps_train + el)
            self.labels_path = config.labels_train
            self.list_IDs = load_list_image_names(config.names_train)
        elif mode == 'val':
            self.image_path = config.dataset_val
            self.segmentation_classid_paths = config.segmentation_classid_val + weather
            # self.segmentation_classid_paths = []
            # self.segmentation_colormap_paths = []
            # for el in weather:
            #     self.segmentation_classid_path.append(config.segmentation_classid_val + el)
            #     self.segmentation_colormap_path.append(config.segmentation_colormaps_val + el)
            self.labels_path = config.labels_val
            self.list_IDs = load_list_image_names(config.names_val)
        # elif mode == 'test':
        #     self.image_path = config.dataset_test
        #     self.list_IDs = load_list_image_names(config.names_test)
        else:
            raise 'ERROR: When calling the dataset, the mode has to be one of train, test, and val.'

        # this function takes up to 18GB of memory temporarily
        dataset_labels = load(self.labels_path, 2)

        self.weather_labels = {}

        # all files in the images_dataset_path
        all_available_images = [f for f in listdir(
            self.image_path) if isfile(join(self.image_path, f))]
        available_ids = []

        def add_to_labels(frame):
            """
            Appends to the dictionary self.weather_labels the element (name_of_the_image: weather_label), if the label
            name is both in self.list_IDs and in all_available_images.
            """
            if frame.name[:-4] in self.list_IDs and frame.name in all_available_images:
                self.weather_labels[frame.name[:-4]] = weather_label_to_id[frame.attributes["weather"]]
                available_ids.append(frame.name[:-4])

        # Go through all of datasets ids and check for their existence in the self.image_path
        # if they are in the path and in our list_IDs then we can add the image to self
        # for label_frame in dataset_labels:
        #     add_to_labels(label_frame)

        list(map(add_to_labels, dataset_labels))


        # print('len(dataset_labels)= ' + str(len(dataset_labels)) + ', len(all_available_images)= ' +
        #       str(len(all_available_images)) + ', len(available_ids)= ' + str(len(available_ids)))

        # update list_IDs with only the available IDs
        self.list_IDs = available_ids

        # Filter out all the images with weather I'm not interested in
        self.list_IDs_filtered = []
        weather_id = weather_label_to_id[self.weather]
        self.length = 0
        for img_id in self.weather_labels:
            if weather_id == self.weather_labels[img_id]:
                self.list_IDs_filtered.append(img_id)
                self.length += 1

        if self.keep_in_memory:
            self.rgb = []
            self.segm_classids = []

            for id in self.list_IDs_filtered:
                self.rgb.append(self.load_image(id))
                self.segm_classids.append(torch.from_numpy(self.load_semantic_classids(id)).to(torch.int64))

            # Transform classids to one-hot encoding
            self.segm_onehot = []
            for el in self.segm_classids:
                onehot = one_hot(el, num_classes=19)
                self.segm_onehot.append(onehot.permute(2, 0, 1))


    def load_image(self, image_name: str) -> np.ndarray:
        return self.load_rgb_image_file(os.path.join(self.image_path, image_name + ".jpg"))

    def load_semantic_classids(self, image_name: str) -> np.ndarray:
        return self.load_semantic_classids_file(os.path.join(self.segmentation_classid_paths, image_name + "_seg_class.png"))

    def load_semantic_classids_file(self, path: Union[str]) -> np.ndarray:
        classid_frame = Image.open(path)
        return np.array(classid_frame)

    def load_rgb_image_file(self, path: Union[str]) -> np.ndarray:
        im_frame = Image.open(path).convert("RGB")
        np_frame = np.array(im_frame)
        return np_frame

    # def semantic_rgb_to_grayscale_map(self, image):
    #     semantic_map = torch.zeros(
    #         [image.shape[0], image.shape[1]], dtype=torch.float32)
    #     for x in range(image.shape[1]):
    #         for y in range(image.shape[0]):
    #             semantic_map[y, x] = self.rgb_to_id(image[y, x])
    #     return semantic_map

    # def rgb_to_id(self, rgb_pixel):
    #     for label in semantic_labels:
    #         if rgb_pixel == label[7]:
    #             return label[1]

    def apply_data_augmentation(self, img):
        img = T.ToTensor()(img)   # transforms to tensor and normalizes to [0, 1]
        img = self.transforms(img)
        # if random.uniform(0, 1) < self.pat:
        #     img = adjust_brightness_random(img)
        # if random.uniform(0, 1) < self.pat:
        #     img = adjust_contrast_random(img)
        # if random.uniform(0, 1) < self.pat:
        #     img = adjust_hue_random(img)
        # if random.uniform(0, 1) < self.pat:
        #     img = adjust_saturation_random(img)
        # if random.uniform(0, 1) < self.pat:
        #     img = adjust_sharpness_random(img)

        return img

    def apply_data_augmentation_to_both(self, img, map):
        img = T.ToTensor()(img)   # transforms to tensor and normalizes to [0, 1]

        # resize
        img = self.resize_img(img)
        map = self.resize_map(map)

        # crop the same way
        #print(img.shape, map.shape, torch.concat((img, map)).shape)
        croped = self.random_crop(torch.concat((img, map)))

        # if torch.rand(1) < 1:
        #   print("flip")
        #   img = torchvision.transforms.functional.hflip(img)
        #   map = torchvision.transforms.functional.hflip(map)
        croped = self.horizontal_flip(croped)
        img = croped[:3, :, :]
        map = croped[3:, :, :]
        # print("flipped", fliped.shape)

        # print("img", img.shape)
        # print("map ", map.shape)

        return img, map


    def __len__(self):
        """Denotes the number of samples in the dataset"""
        return self.length

    def __getitem__(self, index):
        """
        Generates one sample of data. A sample contains (X, label)
        - X is (channels, height, width), where the first 3 channels are the rgb image and the remaining 19 channels
        are the segmentation image - for a total of 22 channels
        - label is the weather_id associated to the called dataloader (e.g. if the dataloader is called with
        weather='snowy', then label=weather_label_to_id['snowy'].
        """
        if self.keep_in_memory:
            rgb_el = self.apply_data_augmentation(self.rgb[index])
            image_semantic = self.segm_onehot[index]
        else:
            rgb_el = self.load_image(self.list_IDs_filtered[index])
            semantic = torch.from_numpy(self.load_semantic_classids(self.list_IDs_filtered[index])).to(torch.int64)
            onehot = one_hot(semantic, num_classes=19)
            image_semantic = onehot.permute(2, 0, 1)

            rgb_el, image_semantic = self.apply_data_augmentation_to_both(rgb_el, image_semantic)

        return torch.cat([rgb_el, image_semantic], dim=0)


# test dataset
if __name__ == '__main__':
    config = Config()
    dataset = Bdd100kDataset(config, mode='val', weather='snowy', keep_in_memory=False)
    img, label = dataset.__getitem__(0)
    print('done')
