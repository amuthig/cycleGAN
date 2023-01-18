import nibabel as nib
import numpy as np
import torch

img = nib.load('data/philips/CC0001_philips_15_55_M.nii.gz')
img2 = nib.load('data/philips/CC0003_philips_15_63_F.nii.gz')

img_tensor = torch.from_numpy(img.get_fdata())
img2_tensor = torch.from_numpy(img2.get_fdata())

print(img_tensor.shape)
print(img2_tensor.shape)