import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import nibabel as nib
from generator_model import Generator
from discriminator_model import Discriminator
import config
import utils

#test du model sur une image issue du dataset au format nifti

disc_A = Discriminator(in_channels=1).to(config.DEVICE) #instanciation du discriminateur
disc_B = Discriminator(in_channels=1).to(config.DEVICE) #instanciation du discriminateur
gen_A2B = Generator(img_channels=1, num_residuals=9).to(config.DEVICE) #instanciation du générateur
gen_B2A = Generator(img_channels=1, num_residuals=9).to(config.DEVICE) #instanciation du générateur
opt_disc = torch.optim.Adam( #instanciation de l'optimiseur
    list(disc_A.parameters()) + list(disc_B.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)
opt_gen = torch.optim.Adam( #instanciation de l'optimiseur
    list(gen_A2B.parameters()) + list(gen_B2A.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)

utils.load_checkpoint( #chargement des poids du model
    config.CHECKPOINT_GEN_B, gen_A2B, opt_gen, config.LEARNING_RATE,
)

gen_A2B.eval() #passage du model en mode evaluation
gen_B2A.eval() #passage du model en mode evaluation



#chargement du volume du même type que celui utilisé pour l'entrainement
volume = nib.load('data/MRI3/TestVolumes/CC0168_siemens_15_38_M.nii.gz')

#chargement du volume issu du style MRI4: 
volume2 = nib.load('data/MRI4/TestVolumes/CC0228_siemens_3_66_M.nii.gz')


#transformation du volume en array
volume = volume.get_fdata()
volume2 = volume2.get_fdata()

#transformation du volume en tensor
volume = torch.from_numpy(volume)
volume2 = torch.from_numpy(volume2)

#exctraction d'une image du volume
idx_image_extraite = 104
image = volume[idx_image_extraite,:,:]
image2 = volume2[idx_image_extraite,:,:]

#transformation de l'image en float
image = image.float()
image2 = image2.float()

#passage de l'image dans le model
image_tmp = image.unsqueeze(0)
image_tmp = image_tmp.unsqueeze(0)
image_tmp = image_tmp.to(config.DEVICE)
fake_B = gen_A2B(image_tmp)
fake_B = fake_B.squeeze(0)
fake_B = fake_B.squeeze(0)

#transformation du tensor en array
image = image.cpu().detach().numpy()
fake_B = fake_B.cpu().detach().numpy()
image2 = image2.cpu().detach().numpy()


#affichage des 3 images sur une même figure avec les légendes
fig, ax = plt.subplots(1,3)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Image originale(MRI3)')
ax[1].imshow(fake_B, cmap='gray')
ax[1].set_title('Image transformée')
ax[2].imshow(image2, cmap='gray')
ax[2].set_title('Style de référence (MRI4)')
plt.show()
