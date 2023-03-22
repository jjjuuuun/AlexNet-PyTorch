import json
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import numpy as np
import zipfile
from tqdm import tqdm
import os
from base import resize, center_crop


def read_json(filename):
    filename = Path(filename)
    with filename.open('r') as f:
        return json.load(f, object_hook=OrderedDict)
    

def preprocess_img(img_path):
    img = Image.open(img_path).convert('RGB')
    # img = resize(img)
    # img = center_crop(img)
    return img


def read_zip(filename):
    imgs = []
    labels = []
    with zipfile.ZipFile(filename, 'r') as f:
        for f_name in tqdm(sorted(f.namelist())):
            img_path = f.open(f_name)
            try:
                imgs.append(preprocess_img(img_path))
                if f_name.find('cat') != -1:
                    labels.append([0])
                else:
                    labels.append([1])
            except:
                print(f"\nNot loading {f_name}")
    return imgs, labels


def read_img(dirname):
    imgs = []
    labels = []
    for filename in tqdm(sorted(os.listdir(dirname))):
        img_path = os.path.join(dirname, filename)
        try:
            imgs.append(preprocess_img(img_path))
            if filename.find('cat') != -1:
                labels.append([0])
            else:
                labels.append([1])
        except:
            print(f"\nNot loading {filename}")
    return imgs, labels

