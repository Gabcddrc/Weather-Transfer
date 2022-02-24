import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class BDD100k(data.Dataset):
    """BDD100k <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    # https://github.com/bdd100k/bdd100k/label.py
    BDD100kClass = namedtuple('BDD100kClass', ['name', 'id', 'trainId', 'category', 'categoryId',
                                                     'hasInstances', 'ignoreInEval', 'color'])
    classes = [
        BDD100kClass("unlabeled",      0, 255, "void", 0, False, True, (0, 0, 0)),
        BDD100kClass("dynamic",        1, 255, "void", 0, False, True, (111, 74, 0)),
        BDD100kClass("ego vehicle",    2, 255, "void", 0, False, True, (0, 0, 0)),
        BDD100kClass("ground",         3, 255, "void", 0, False, True, (81, 0, 81)),
        BDD100kClass("static",         4, 255, "void", 0, False, True, (0, 0, 0)),
        BDD100kClass("parking",        5, 255, "flat", 1, False, True, (250, 170, 160)),
        BDD100kClass("rail track",     6, 255, "flat", 1, False, True, (230, 150, 140)),
        BDD100kClass("road",           7, 0, "flat", 1, False, False, (128, 64, 128)),
        BDD100kClass("sidewalk",       8, 1, "flat", 1, False, False, (244, 35, 232)),
        BDD100kClass("bridge",         9, 255, "construction", 2, False, True, (150, 100, 100)),
        BDD100kClass("building",       10, 2, "construction", 2, False, False, (70, 70, 70)),
        BDD100kClass("fence",          11, 4, "construction", 2, False, False, (190, 153, 153)),
        BDD100kClass("garage",         12, 255, "construction", 2, False, True, (180, 100, 180)),
        BDD100kClass("guard rail",     13, 255, "construction", 2, False, True, (180, 165, 180)),
        BDD100kClass("tunnel",         14, 255, "construction", 2, False, True, (150, 120, 90)),
        BDD100kClass("wall",           15, 3, "construction", 2, False, False, (102, 102, 156)),
        BDD100kClass("banner",         16, 255, "object", 3, False, True, (250, 170, 100)),
        BDD100kClass("billboard",      17, 255, "object", 3, False, True, (220, 220, 250)),
        BDD100kClass("lane divider",   18, 255, "object", 3, False, True, (255, 165, 0)),
        BDD100kClass("parking sign",   19, 255, "object", 3, False, False, (220, 20, 60)),
        BDD100kClass("pole",           20, 5, "object", 3, False, False, (153, 153, 153)),
        BDD100kClass("polegroup",      21, 255, "object", 3, False, True, (153, 153, 153)),
        BDD100kClass("street light",   22, 255, "object", 3, False, True, (220, 220, 100)),
        BDD100kClass("traffic cone",   23, 255, "object", 3, False, True, (255, 70, 0)),
        BDD100kClass("traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)),
        BDD100kClass("traffic light",  25, 6, "object", 3, False, False, (250, 170, 30)),
        BDD100kClass("traffic sign",   26, 7, "object", 3, False, False, (220, 220, 0)),
        BDD100kClass("traffic sign frame",27,255,"object",3,False,True,(250, 170, 250),),
        BDD100kClass("terrain",        28, 9, "nature", 4, False, False, (152, 251, 152)),
        BDD100kClass("vegetation",     29, 8, "nature", 4, False, False, (107, 142, 35)),
        BDD100kClass("sky",            30, 10, "sky", 5, False, False, (70, 130, 180)),
        BDD100kClass("person",         31, 11, "human", 6, True, False, (220, 20, 60)),
        BDD100kClass("rider",          32, 12, "human", 6, True, False, (255, 0, 0)),
        BDD100kClass("bicycle",        33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        BDD100kClass("bus",            34, 15, "vehicle", 7, True, False, (0, 60, 100)),
        BDD100kClass("car",            35, 13, "vehicle", 7, True, False, (0, 0, 142)),
        BDD100kClass("caravan",        36, 255, "vehicle", 7, True, True, (0, 0, 90)),
        BDD100kClass("motorcycle",     37, 17, "vehicle", 7, True, False, (0, 0, 230)),
        BDD100kClass("trailer",        38, 255, "vehicle", 7, True, True, (0, 0, 110)),
        BDD100kClass("train",          39, 16, "vehicle", 7, True, False, (0, 80, 100)),
        BDD100kClass("truck",          40, 14, "vehicle", 7, True, False, (0, 0, 70)),
    ]

    train_id_to_color = [c.color for c in classes if (c.trainId != -1 and c.trainId != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.trainId for c in classes])

    def __init__(self, root, split='train', target_type='colormaps', transform=None):
        self.root = os.path.expanduser(root)
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'images', '10k', split)

        self.targets_dir = os.path.join(self.root, 'labels', 'sem_seg', target_type, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for target_filename in os.listdir(self.targets_dir):
            image_filename = target_filename[:-4] + '.jpg'
            # possible_images = [
            # os.path.join(self.images_dir, 'train', image_filename),
            # os.path.join(self.images_dir, 'val', image_filename),
            # os.path.join(self.images_dir, 'test', image_filename)
            # ]
            imn = os.path.join(self.images_dir, image_filename)
            #for imn in possible_images:
            if os.path.isfile(imn):
                self.images.append(imn) # full path
                self.targets.append(os.path.join(self.targets_dir, target_filename))
                #break
                # only append those exist in images

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        #print(target.numpy())
        #target = self.encode_target(target) # already trainID
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
