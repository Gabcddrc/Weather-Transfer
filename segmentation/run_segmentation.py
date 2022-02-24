from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os, sys
import random
import argparse
import numpy as np
import pandas as pd

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
#from torchinfo import summary
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

PATH_root = '/psi/home/li_s1/data/Season'
PATH_model = '/psi/home/li_s1/data/Season/pretrained'
PATH_data = '/psi/home/li_s1/data/Season/images_400_224/100k'
PATH_name = '/psi/home/li_s1/DeepLearning2021/dataset/names'
PATH_seg = '/psi/home/li_s1/data/Season/classids_400_224_better'

train_path = os.path.join(PATH_data, "train")
val_path = os.path.join(PATH_data, "val")
test_path = os.path.join(PATH_data, "test")
data_set_paths = {'train': train_path, 'val': val_path, 'test': test_path}
# data_set_paths = [train_path, val_path, test_path]
# data_set_paths = [os.path.normpath(path) for path in data_set_paths]

df_names_weather = pd.read_csv(os.path.join(PATH_root, 'both_name_weather.csv'))

def segmentation(split, weather, ind, ind_total=5):
    # Setup dataloader
    input_path = data_set_paths[split]
    image_files = []
    for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
        files = glob(os.path.join(input_path, '**/*.%s'%(ext)), recursive=True)
        if len(files)>0:
            image_files.extend(files)
    print(len(image_files))
    names_weather = df_names_weather[(df_names_weather['weather']==weather) & (df_names_weather['dataset']==split)].name.values
    names_weather = [os.path.join(input_path, n) for n in names_weather]
    print(len(names_weather))
    names_final = list(set(names_weather).intersection(set(image_files)))
    names_final.sort()
    print(len(names_final))

    N = len(names_final)
    if int(N/ind_total*(ind+1)) >= N:
        names_final = names_final[int(N/ind_total*ind):]
        print('start:', int(N/ind_total*ind))
    else:
        names_final = names_final[int(N/ind_total*ind):int(N/ind_total*(ind+1))]
        print('start:', int(N/ind_total*ind), 'end:', int(N/ind_total*(ind+1)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    CKPT_PATH = os.path.join(PATH_model, 'latest_deeplabv3plus_mobilenet_BDD100k_os16.pth')

    # Set up model
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    del checkpoint

    decode_fn = Cityscapes.decode_target
    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                ])

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(names_final):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            classid_preds = Image.fromarray(pred.astype('uint8'))
            # colorized_preds = decode_fn(pred).astype('uint8')
            # colorized_preds = Image.fromarray(colorized_preds)

            classid_preds.save(os.path.join(PATH_seg, split, weather, img_name+'_seg_class.png'))
            # colorized_preds.save(os.path.join(PATH_seg, 'colormaps', split, weather, img_name+'_seg_color.png'))

if __name__ == '__main__':
    segmentation(split=sys.argv[1], weather=sys.argv[2], ind=int(sys.argv[3]), ind_total=int(sys.argv[4]))
