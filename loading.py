import os
from glob import glob

import numpy as np
import nibabel as nib
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism






def prepare(in_dir, MRI, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128,128,128], cache=False ):

    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you 
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, str(MRI), "*.nii.gz")))


    train_files = [{"vol": image_name} for image_name in zip(path_train_volumes)]


    train_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            AddChanneld(keys=["vol"]),
            # Spacingd(keys=["vol"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=["vol"], source_key="vol"),
            Resized(keys=["vol"], spatial_size=spatial_size),   
            ToTensord(keys=["vol"]),

        ]
    )

    

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        
        return train_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)
       
        return train_loader
