import torch
import torch.nn as nn
from tqdm import tqdm
import config
from torchvision.utils import save_image
from dataset import HorseZebrasDataset
from utils import load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader

from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_A, disc_B, gen_A2B, gen_B2A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True) #progress bar

    for idx, (real_A, real_B) in enumerate(loop):
        real_A = real_A.to(config.DEVICE)
        real_B = real_B.to(config.DEVICE)

        # Train Discriminators: max log(D(x)) + log(1 - D(G(z)))
        # where D(x) is the probability that the discriminator
        # correctly classifies the real image as real
        # and D(G(z)) is the probability that the discriminator
        # correctly classifies the fake image as fake
        with torch.cuda.amp.autocast():
            fake_B = gen_A2B(real_A)
            disc_real_B = disc_B(real_B)
            disc_fake_B = disc_B(fake_B)
            disc_B_loss = (
                mse(disc_real_B, torch.ones_like(disc_real_B))
                + mse(disc_fake_B, torch.zeros_like(disc_fake_B))
            )

            fake_A = gen_B2A(real_B)
            disc_real_A = disc_A(real_A)
            disc_fake_A = disc_A(fake_A)
            disc_A_loss = (
                mse(disc_real_A, torch.ones_like(disc_real_A))
                + mse(disc_fake_A, torch.zeros_like(disc_fake_A))
            )

            disc_loss = (disc_A_loss + disc_B_loss)/2

        opt_disc.zero_grad() #zero the gradient buffers
        d_scaler.scale(disc_loss).backward() #backward pass
        d_scaler.step(opt_disc) #update the weights
        d_scaler.update() #update the scale for next iteration

        # Train Generators: min log(1 - D(G(z))) <-> max log(D(G(z))
        # Train Generators: min L1(y, G(x)) <-> max L1(y, G(x))
        with torch.cuda.amp.autocast():
            # adversarial loss is binary cross-entropy
            fake_B = gen_A2B(real_A)
            disc_fake_B = disc_B(fake_B)
            gen_A2B_loss = mse(disc_fake_B, torch.ones_like(disc_fake_B)) #mse = mean squared error

            fake_A = gen_B2A(real_B)
            disc_fake_A = disc_A(fake_A)
            gen_B2A_loss = mse(disc_fake_A, torch.ones_like(disc_fake_A)) #mse = mean squared error

            # cycle loss
            recov_A = gen_B2A(fake_B)
            cycle_A_loss = L1(real_A, recov_A)

            recov_B = gen_A2B(fake_A)
            cycle_B_loss = L1(real_B, recov_B)

            # identity loss
            #id_A = gen_B2A(real_A)
            #id_A_loss = L1(real_A, id_A)

            #id_B = gen_A2B(real_B)
            #id_B_loss = L1(real_B, id_B) 

            gen_loss = ( #loss of the generator
                gen_B2A_loss
                + gen_A2B_loss
                + cycle_A_loss*config.LAMBDA_CYCLE
                + cycle_B_loss*config.LAMBDA_CYCLE
                #+ id_A_loss
                #+ id_B_loss
            )
        opt_gen.zero_grad() #zero the gradient buffers
        g_scaler.scale(gen_loss).backward() #backward pass
        g_scaler.step(opt_gen) #update the weights
        g_scaler.update() #update the scale for next iteration

        if idx % 100 == 0:
            save_image(fake_A*0.5 + 0.5, f"saved_images/{idx}_fake_A.png")
            save_image(fake_B*0.5 + 0.5, f"saved_images/{idx}_fake_B.png")







def main():
    print("coucou")
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
    

    dataset = HorseZebrasDataset(root_horse=config.TRAIN_DIR + "/horses", root_zebra=config.TRAIN_DIR + "/zebras", transform=config.transforms) #create the dataset

    loader = torch.utils.data.DataLoader( #create the dataloader
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler() #create the gradient scaler for the generator
    d_scaler = torch.cuda.amp.GradScaler() #create the gradient scaler for the discriminator

    for epoch in range(config.NUM_EPOCHS): #loop through the epochs
        train_fn(disc_A, disc_B, gen_A2B, gen_B2A, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL: #save the model
            save_checkpoint(gen_A2B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(gen_B2A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

if __name__ == "__main__":
    main()