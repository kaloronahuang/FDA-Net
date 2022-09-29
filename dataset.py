# dataset.py
import torch, os, json
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.io import read_image

class FDAImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.embeddings = json.load(open(annotations_file, 'r'))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.file_list = list(self.embeddings.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_list[idx])
        image = read_image(img_path)
        label = np.array([[self.embeddings[self.file_list[idx]], ], ], dtype=np.float32).transpose((2, 0, 1))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label