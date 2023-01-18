import torch
import torch.nn as nn
from tqdm import tqdm
import config
from torchvision.utils import save_image

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