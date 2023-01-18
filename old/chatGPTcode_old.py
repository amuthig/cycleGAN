import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import nibabel as nib
from loading_old import prepare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the layers of the generator here
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.conv6 = nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up5 = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        # Define the forward pass of the generator here
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.up1(F.relu(self.bn2(self.conv2(x))))
        x = self.up2(F.relu(self.bn3(self.conv3(x))))
        x = self.up3(F.relu(self.bn4(self.conv4(x))))
        x = self.up4(F.relu(self.bn5(self.conv5(x))))
        x = self.up5(self.conv6(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.conv6 = nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = x.view(-1, 1)
        return x



# Define the loss functions
criterion_G = nn.MSELoss()
criterion_D = nn.BCEWithLogitsLoss()
learning_rate = 0.1 
# Define the optimizers
optimizer_G = optim.Adam(Generator().parameters(), lr=learning_rate)
optimizer_D = optim.Adam(Discriminator().parameters(), lr=learning_rate)



import nibabel as nib
from torchvision.transforms import Resize

# class NiftiDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.file_list = os.listdir(self.root_dir)
#         self.resize = Resize((128, 128,128))

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.root_dir, self.file_list[idx])
#         image = nib.load(file_path).get_fdata()
#         if self.transform:
#             image = self.transform(image)
#         image = self.resize(image)
#         return image



# # Create the data loaders
# style_A_dataset = NiftiDataset(root_dir='H:\TDSI\cc359_preprocessed\TrainVolumes', transform=transforms.ToTensor())
# style_B_dataset = NiftiDataset(root_dir='H:\TDSI\cc359_preprocessed\Different\TrainVolumes', transform=transforms.ToTensor())

batch_size = 4 
# data_loader_A = DataLoader(style_A_dataset, batch_size=batch_size, shuffle=True)
# data_loader_B = DataLoader(style_B_dataset, batch_size=batch_size, shuffle=True)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
data = 'H:\TDSI\cc359_preprocessed'
source = "MRI1"
cible = "MRI4"

data_loader_A = prepare(data,MRI=source, cache=True)
data_loader_B = prepare(data,MRI=cible, cache=True)


# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()
num_epochs = 100
lambda_cycle = 10
# Train the model
for epoch in range(num_epochs):
    for i, (style_A, style_B) in enumerate(zip(data_loader_A, data_loader_B)):
        # Train the discriminator
        optimizer_D.zero_grad()
        style_A = data_loader_A["vol"]
        style_B = data_loader_B["vol"]



        real_A = style_A.to(device)
        real_B = style_B.to(device)
        fake_B = generator(real_A).detach()
        fake_A = generator(real_B).detach()
        loss_D_real = criterion_D(discriminator(real_A), torch.ones(batch_size, 1).to(device))
        loss_D_fake = criterion_D(discriminator(fake_A), torch.zeros(batch_size, 1).to(device))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        fake_A = generator(real_B)
        loss_G_A = criterion_G(fake_A, real_A)
        loss_G_B = criterion_G(fake_B, real_B)
        loss_cycle_A = criterion_G(generator(fake_B), real_A)
        loss_cycle_B = criterion_G(generator(fake_A), real_B)
        loss_G = loss_G_A + loss_G_B + lambda_cycle * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()

    # Print the current loss values
    print("Epoch [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}".format(epoch+1, num_epochs, loss_D.item(), loss_G.item()))

# Save the trained model
torch.save(generator.state_dict(), "cycle_gan_generator.pth")
torch.save(discriminator.state_dict(), "cycle_gan_discriminator.pth")

