import sys
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, img_width, img_height):
        self.image_dir      = image_dir
        self.img_width      = img_width
        self.img_height     = img_height
        self.images         = sorted(os.listdir(image_dir))

        self.dim = (img_width, img_height)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_img = (cv2.resize(img, self.dim) / 255).transpose((2, 0, 1))

        x = torch.Tensor(original_img)

        return x, x
