import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
import os
import nibabel as nib

class ResnetBlock(nn.Module):
    """Residual block for ResnetGenerator"""
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Resnet-based generator"""
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int):
        super().__init__()

        # Initial convolution block       
        model = [   nn.Conv3d(input_nc, 64, kernel_size=7, stride=1, padding=3),
                    nn.InstanceNorm3d(64),
                    nn.ReLU(inplace=True) ]
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv3d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose3d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        # Output layer
        model += [nn.Conv3d(64, output_nc, kernel_size=7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

    
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN"""
    def __init__(self, input_nc: int):
        super().__init__()

        model = [   nn.Conv3d(input_nc, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.01) ]

        model += [  nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm3d(128), 
                    nn.LeakyReLU(0.01) ]

        model += [  nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm3d(256), 
                    nn.LeakyReLU(0.01) ]

        model += [  nn.Conv3d(256, 512, kernel_size=4, stride=1, padding=1),
                    nn.InstanceNorm3d(512), 
                    nn.LeakyReLU(0.01) ]

        model += [nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return x.view(x.size(0), -1)

class CycleGAN(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int):
        super().__init__()

        # Initialize generator and discriminator
        self.gen_A = ResnetGenerator(input_nc, output_nc, n_residual_blocks)
        self.gen_B = ResnetGenerator(input_nc, output_nc, n_residual_blocks)
        self.dis_A = Discriminator(input_nc)
        self.dis_B = Discriminator(input_nc)

    def forward(self, real_A, real_B):
        # Translate images
        fake_B = self.gen_A(real_A)
        fake_A = self.gen_B(real_B)
        # Translate back to original domain
        rec_A = self.gen_B(fake_B)
        rec_B = self.gen_A(fake_A)
        # Discriminator output
        dis_real_A = self.dis_A(real_A)
        dis_real_B = self.dis_B(real_B)
        dis_fake_A = self.dis_A(fake_A)
        dis_fake_B = self.dis_B(fake_B)

        return fake_A, fake_B, rec_A, rec_B, dis_real_A, dis_real_B, dis_fake_A, dis_fake_B

# Define dataset and datal

class MedicalImageDataset(BaseDataset):
    def __init__(self, data_folder: str, transform: transforms.Compose):
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_folder))

    def __getitem__(self, index: int):
        img = nib.load(os.path.join(self.data_folder, str(index) + '.nii')).get_fdata()
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        return img



   # Define dataset and dataloader for domain A and B
transform = transforms.Compose([transforms.ToTensor()])
dataset_A = MedicalImageDataset('H:\TDSI\cc359_preprocessed\TrainVolumes', transform)
dataloader_A = DataLoader(dataset_A, batch_size=4, shuffle=True)

dataset_B = MedicalImageDataset('H:\TDSI\cc359_preprocessed\Different\TrainVolumes', transform)
dataloader_B = DataLoader(dataset_B, batch_size=4, shuffle=True)

# Initialize generator
cycle_gan = CycleGAN(input_nc=1, output_nc=1, n_residual_blocks=9)

# Set generator to evaluation mode
cycle_gan.eval()

# Generate images and save them
with torch.no_grad():
    for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
        fake_A, _, _, _, _, _, _, _ = cycle_gan(real_A, real_B)
        for j in range(fake_A.size(0)):
            img = fake_A[j, :, :, :].squeeze().numpy()
            img = (img + 1) * 127.5
            img = img.astype(np.uint8)
            # Save the generated image
            img = nib.Nifti1Image(img, np.eye(4))
            nib.save(img, f'H:\TDSI\cc359_preprocessed\Sortie/{i*4+j}.nii')
    print("All files generated!")

