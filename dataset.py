"""
import nibabel as nib
import numpy as np
import torch

vol1 = nib.load('data/MRI2/')
vol2 = nib.load('data/philips/CC0003_philips_15_63_F.nii.gz')

img_tensor = torch.from_numpy(img.get_fdata())
img2_tensor = torch.from_numpy(img2.get_fdata())

print(img_tensor.shape)
print(img2_tensor.shape)
"""

from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class HorseZebrasDataset(Dataset):
    def __init__(self, root_horse, root_zebra, transform=None):
        self.root_horse = root_horse
        self.root_zebra = root_zebra
        self.transform = transform
        self.horse_images = os.listdir(root_horse)
        self.zebra_images = os.listdir(root_zebra)

        #on ne va pas utiliser toutes les images, on va prendre le minimum des 2 datasets
        self.length_dataset = min(len(self.horse_images), len(self.zebra_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        horse_img = self.horse_images[index]
        zebra_img = self.zebra_images[index]
        
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img