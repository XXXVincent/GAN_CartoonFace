from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class FaceDataset(Dataset):
    """Face dataset."""

    def __init__(self, dataset_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.img_list = os.listdir(self.dataset_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.dataset_dir,
                               self.img_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image




def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = sample_batched
    batch_size = len(images_batch)
    plt.figure()
    for i in range(batch_size):
        img = sample_batched[i].numpy()
        plt.title('Batch from dataloader')
        plt.imshow(img)
        plt.show()

# if __name__ == '__main__':
#     dataloader = DataLoader(FaceDataset(dataset_dir=r'C:\Users\vincent.xu\Desktop\BD_DL\test'), batch_size=4,
#                             shuffle=True)
#     for i_batch, sample_batched in enumerate(dataloader):
#         print(i_batch, sample_batched.size())
#         show_batch(sample_batched)
