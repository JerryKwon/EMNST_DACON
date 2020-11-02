# emnst_dataset.py
# custom pytorch Dataset for emnst dataset

import numpy as np
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset

class EMNST_Dataset(Dataset):
    def __init__(self, img_dict, img_height, img_width, transform):
        self.img_dict = img_dict
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, index):
        img_dict = self.img_dict[index]
        img_id = img_dict["img_id"]
        img = img_dict["img"].reshape(self.img_height, self.img_width)
        digit = img_dict["digit"]
        letter = img_dict["letter"]
        img = img.astype("uint8")

        if self.transform is not None:
            img = self.transform(img)

        digit = np.uint8(digit)

        return img, torch.tensor(digit, dtype=torch.long)

class EMNST_Test_Dataset(Dataset):
    def __init__(self, img_dict, img_height, img_width, transform):
        self.img_dict = img_dict
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, index):
        img_dict = self.img_dict[index]
        img_id = int(img_dict["img_id"])
        img = img_dict["img"].reshape(self.img_height, self.img_width)
        letter = img_dict["letter"]
        img = img.astype("uint8")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(img_id, dtype=torch.long)