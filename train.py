import torch
import torch.nn as nn
from tqdm import tqdm
import config
from torchvision.utils import save_image
from utils import load_checkpoint, save_checkpoint

from discriminator_model import Discriminator
from generator_model import Generator

def train_fn():
    pass

def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_A2B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_B2A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = torch.optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = torch.optim.Adam(
        list(gen_A2B.parameters()) + list(gen_B2A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A2B, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_B2A, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE,
        )
    
    #A faire: créer dataloader, dans dataset.py, l'instancier ici