# cyclegan_windows_safe.py
# Code credit : PyTorch-GAN Git repo

import os
import sys
import time
import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from models import *
from datasets import *
from utils import *
import torch.multiprocessing as mp


def main():
    # -----------------------
    # Parameters
    # -------------
    start_epoch = 7  # change if you want to continue training
    n_epochs = 20
    dataset_name = "sar_optical_image"
    batch_size = 2
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    decay_epoch = 10
    n_cpu = 16
    img_height, img_width, channels = 256, 256, 3
    sample_interval = 500
    n_residual_blocks = 5
    lambda_cyc = 10
    lambda_id = 5.0
    checkpoint_interval = 1
    val_interval = 2  # validation every 2 epochs
    plot_interval = 2  # plot graphs every 2 epochs

    # Create directories
    os.makedirs(f"images/{dataset_name}", exist_ok=True)
    os.makedirs(f"saved_models/{dataset_name}", exist_ok=True)
    os.makedirs("images/graphs", exist_ok=True)  # Create graphs directory

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    input_shape = (channels, img_height, img_width)

    # -----------------------
    # Initialize models
    # -----------------------
    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    if cuda:
        G_AB, G_BA, D_A, D_B = G_AB.cuda(), G_BA.cuda(), D_A.cuda(), D_B.cuda()
        criterion_GAN, criterion_cycle, criterion_identity = criterion_GAN.cuda(), criterion_cycle.cuda(), criterion_identity.cuda()

    # Load weights if continuing
    if start_epoch != 0:
        G_AB.load_state_dict(torch.load(f"saved_models/{dataset_name}/G_AB_{start_epoch}.pth"))
        G_BA.load_state_dict(torch.load(f"saved_models/{dataset_name}/G_BA_{start_epoch}.pth"))
        D_A.load_state_dict(torch.load(f"saved_models/{dataset_name}/D_A_{start_epoch}.pth"))
        D_B.load_state_dict(torch.load(f"saved_models/{dataset_name}/D_B_{start_epoch}.pth"))
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # -----------------------
    # Optimizers & schedulers
    # -----------------------
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

    # -----------------------
    # Dataloaders
    # -----------------------
    transforms_ = [
        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_loader = DataLoader(
        ImageDataset(f"data/{dataset_name}", transforms_=transforms_, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True
    )

    val_loader = DataLoader(
        ImageDataset(f"data/{dataset_name}", transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True
    )

    # Buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # For plotting losses
    train_G_losses, train_D_losses = [], []
    train_GAN_losses, train_cycle_losses, train_identity_losses = [], [], []
    val_G_losses, val_D_losses = [], []
    val_GAN_losses, val_cycle_losses, val_identity_losses = [], [], []

    prev_time = time.time()

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(start_epoch, n_epochs):
        G_AB.train(); G_BA.train()
        epoch_G_loss, epoch_D_loss = 0, 0
        epoch_GAN_loss, epoch_cycle_loss, epoch_identity_loss = 0, 0, 0

        for i, batch in enumerate(train_loader):
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            valid = Variable(torch.ones((real_A.size(0), *D_A.output_shape), dtype=torch.float32, device=real_A.device), requires_grad=False)
            fake = Variable(torch.zeros((real_A.size(0), *D_A.output_shape), dtype=torch.float32, device=real_A.device), requires_grad=False)

            # Train Generators
            optimizer_G.zero_grad()
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B)/2

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2

            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle = (criterion_cycle(recov_A, real_A) + criterion_cycle(recov_B, real_B))/2

            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminators
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            loss_fake = criterion_GAN(D_A(fake_A_buffer.push_and_pop(fake_A).detach()), fake)
            loss_D_A = (loss_real + loss_fake)/2
            loss_D_A.backward(); optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            loss_fake = criterion_GAN(D_B(fake_B_buffer.push_and_pop(fake_B).detach()), fake)
            loss_D_B = (loss_real + loss_fake)/2
            loss_D_B.backward(); optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B)/2

            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            epoch_GAN_loss += loss_GAN.item()
            epoch_cycle_loss += loss_cycle.item()
            epoch_identity_loss += loss_identity.item()

            batches_done = epoch * len(train_loader) + i
            batches_left = n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(f"\r[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                             f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, adv: {loss_GAN.item():.4f}, "
                             f"cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}] ETA: {time_left}")

            if batches_done % sample_interval == 0:
                # Save sample images
                sample_images(val_loader, G_AB, G_BA, Tensor, dataset_name, batches_done)

        # Average losses
        train_G_losses.append(epoch_G_loss/len(train_loader))
        train_D_losses.append(epoch_D_loss/len(train_loader))
        train_GAN_losses.append(epoch_GAN_loss/len(train_loader))
        train_cycle_losses.append(epoch_cycle_loss/len(train_loader))
        train_identity_losses.append(epoch_identity_loss/len(train_loader))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Validation every val_interval
        if epoch % val_interval == 0:
            val_G, val_D, val_GAN, val_cycle, val_identity = validate(val_loader, G_AB, G_BA, D_A, D_B, Tensor, criterion_GAN, criterion_cycle, criterion_identity, lambda_cyc, lambda_id)
            val_G_losses.append(val_G)
            val_D_losses.append(val_D)
            val_GAN_losses.append(val_GAN)
            val_cycle_losses.append(val_cycle)
            val_identity_losses.append(val_identity)
            print(f"\n[Validation] Epoch {epoch}: G loss: {val_G:.4f}, D loss: {val_D:.4f}, "
                  f"GAN: {val_GAN:.4f}, Cycle: {val_cycle:.4f}, Identity: {val_identity:.4f}")
        
        # Print epoch summary
        print(f"\n[Epoch {epoch} Summary] Train - G: {train_G_losses[-1]:.4f}, D: {train_D_losses[-1]:.4f}, "
              f"GAN: {train_GAN_losses[-1]:.4f}, Cycle: {train_cycle_losses[-1]:.4f}, Identity: {train_identity_losses[-1]:.4f}")

        # Plot graphs every plot_interval epochs
        if (epoch - start_epoch) % plot_interval == 0 and epoch >= start_epoch:
            plot_losses_realtime(train_G_losses, val_G_losses, train_D_losses, val_D_losses, 
                                train_GAN_losses, val_GAN_losses, train_cycle_losses, val_cycle_losses,
                                train_identity_losses, val_identity_losses, start_epoch, epoch, dataset_name, val_interval)

        # Save checkpoint every checkpoint_interval
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            torch.save(G_AB.state_dict(), f"saved_models/{dataset_name}/G_AB_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"saved_models/{dataset_name}/G_BA_{epoch}.pth")
            torch.save(D_A.state_dict(), f"saved_models/{dataset_name}/D_A_{epoch}.pth")
            torch.save(D_B.state_dict(), f"saved_models/{dataset_name}/D_B_{epoch}.pth")

    # -----------------------
    # Plot final train & val losses
    # -----------------------
    plot_losses_final(train_G_losses, val_G_losses, train_D_losses, val_D_losses, 
                     train_GAN_losses, val_GAN_losses, train_cycle_losses, val_cycle_losses,
                     train_identity_losses, val_identity_losses, start_epoch, n_epochs, dataset_name, val_interval)


# -----------------------
# Sample images function
# -----------------------
def sample_images(val_loader, G_AB, G_BA, Tensor, dataset_name, batches_done):
    G_AB.eval(); G_BA.eval()
    imgs = next(iter(val_loader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"images/{dataset_name}/{batches_done}.png", normalize=False)


# -----------------------
# Validation function
# -----------------------
def validate(val_loader, G_AB, G_BA, D_A, D_B, Tensor, criterion_GAN, criterion_cycle, criterion_identity, lambda_cyc, lambda_id):
    G_AB.eval(); G_BA.eval()
    val_G_loss, val_D_loss = 0, 0
    val_GAN_loss, val_cycle_loss, val_identity_loss = 0, 0, 0
    
    with torch.no_grad():
        for batch in val_loader:
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            valid = Variable(torch.ones((real_A.size(0), *D_A.output_shape), dtype=torch.float32, device=real_A.device), requires_grad=False)
            fake = Variable(torch.zeros((real_A.size(0), *D_A.output_shape), dtype=torch.float32, device=real_A.device), requires_grad=False)

            # Generator loss components
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B)/2
            
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2
            
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle = (criterion_cycle(recov_A, real_A) + criterion_cycle(recov_B, real_B))/2
            
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            
            val_G_loss += loss_G.item()
            val_GAN_loss += loss_GAN.item()
            val_cycle_loss += loss_cycle.item()
            val_identity_loss += loss_identity.item()

            # Discriminator loss
            loss_D_A = (criterion_GAN(D_A(real_A), valid) + criterion_GAN(D_A(fake_A.detach()), fake))/2
            loss_D_B = (criterion_GAN(D_B(real_B), valid) + criterion_GAN(D_B(fake_B.detach()), fake))/2
            val_D_loss += (loss_D_A + loss_D_B).item()/2

    # Return as Python floats (already converted with .item() above)
    return (val_G_loss/len(val_loader), val_D_loss/len(val_loader), 
            val_GAN_loss/len(val_loader), val_cycle_loss/len(val_loader), 
            val_identity_loss/len(val_loader))


# -----------------------
# Real-time plotting function (every 3 epochs)
# -----------------------
def plot_losses_realtime(train_G, val_G, train_D, val_D, train_GAN, val_GAN, train_cycle, val_cycle, 
                        train_identity, val_identity, start_epoch, current_epoch, dataset_name, val_interval):
    epochs = range(start_epoch, current_epoch + 1)
    val_epochs = list(range(start_epoch, current_epoch + 1, val_interval))  # validation every val_interval epochs
    
    # Ensure we have validation data
    if len(val_epochs) > len(val_G):
        val_epochs = val_epochs[:len(val_G)]
    
    # Create subplots for all losses
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'CycleGAN Training Losses - Epoch {current_epoch}', fontsize=16)
    
    # Generator Loss
    axes[0,0].plot(epochs, train_G, label="Train Generator", marker='o', markersize=3, color='blue')
    if len(val_G) > 0 and len(val_epochs) > 0:
        axes[0,0].plot(val_epochs, val_G, label="Val Generator", marker='x', markersize=5, color='red')
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Generator Loss")
    axes[0,0].set_title("Total Generator Loss"); axes[0,0].legend(); axes[0,0].grid(True)
    
    # Discriminator Loss
    axes[0,1].plot(epochs, train_D, label="Train Discriminator", marker='o', markersize=3, color='green')
    if len(val_D) > 0 and len(val_epochs) > 0:
        axes[0,1].plot(val_epochs, val_D, label="Val Discriminator", marker='x', markersize=5, color='orange')
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Discriminator Loss")
    axes[0,1].set_title("Discriminator Loss"); axes[0,1].legend(); axes[0,1].grid(True)
    
    # GAN/Adversarial Loss
    axes[0,2].plot(epochs, train_GAN, label="Train GAN", marker='o', markersize=3, color='purple')
    if len(val_GAN) > 0 and len(val_epochs) > 0:
        axes[0,2].plot(val_epochs, val_GAN, label="Val GAN", marker='x', markersize=5, color='brown')
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("GAN Loss")
    axes[0,2].set_title("Adversarial Loss"); axes[0,2].legend(); axes[0,2].grid(True)
    
    # Cycle Loss
    axes[1,0].plot(epochs, train_cycle, label="Train Cycle", marker='o', markersize=3, color='cyan')
    if len(val_cycle) > 0 and len(val_epochs) > 0:
        axes[1,0].plot(val_epochs, val_cycle, label="Val Cycle", marker='x', markersize=5, color='magenta')
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("Cycle Loss")
    axes[1,0].set_title("Cycle Consistency Loss"); axes[1,0].legend(); axes[1,0].grid(True)
    
    # Identity Loss
    axes[1,1].plot(epochs, train_identity, label="Train Identity", marker='o', markersize=3, color='yellow')
    if len(val_identity) > 0 and len(val_epochs) > 0:
        axes[1,1].plot(val_epochs, val_identity, label="Val Identity", marker='x', markersize=5, color='black')
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Identity Loss")
    axes[1,1].set_title("Identity Loss"); axes[1,1].legend(); axes[1,1].grid(True)
    
    # Combined Loss Comparison
    axes[1,2].plot(epochs, train_G, label="Generator", marker='o', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_D, label="Discriminator", marker='s', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_GAN, label="GAN", marker='^', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_cycle, label="Cycle", marker='v', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_identity, label="Identity", marker='d', markersize=2, alpha=0.7)
    axes[1,2].set_xlabel("Epoch"); axes[1,2].set_ylabel("Loss Value")
    axes[1,2].set_title("All Training Losses"); axes[1,2].legend(); axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"images/graphs/losses_epoch_{current_epoch}.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Graphs saved for epoch {current_epoch} in images/graphs/")


# -----------------------
# Final plotting function (at end of training)
# -----------------------
def plot_losses_final(train_G, val_G, train_D, val_D, train_GAN, val_GAN, train_cycle, val_cycle, 
                     train_identity, val_identity, start_epoch, n_epochs, dataset_name, val_interval):
    epochs = range(start_epoch, n_epochs)
    val_epochs = list(range(start_epoch, n_epochs, val_interval))  # validation every val_interval epochs
    
    # Ensure we have validation data
    if len(val_epochs) > len(val_G):
        val_epochs = val_epochs[:len(val_G)]
    
    # Create subplots for all losses
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CycleGAN Final Training Losses', fontsize=16)
    
    # Generator Loss
    axes[0,0].plot(epochs, train_G, label="Train Generator", marker='o', markersize=3)
    if len(val_G) > 0 and len(val_epochs) > 0:
        axes[0,0].plot(val_epochs, val_G, label="Val Generator", marker='x', markersize=5)
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Generator Loss")
    axes[0,0].set_title("Total Generator Loss"); axes[0,0].legend(); axes[0,0].grid(True)
    
    # Discriminator Loss
    axes[0,1].plot(epochs, train_D, label="Train Discriminator", marker='o', markersize=3)
    if len(val_D) > 0 and len(val_epochs) > 0:
        axes[0,1].plot(val_epochs, val_D, label="Val Discriminator", marker='x', markersize=5)
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Discriminator Loss")
    axes[0,1].set_title("Discriminator Loss"); axes[0,1].legend(); axes[0,1].grid(True)
    
    # GAN/Adversarial Loss
    axes[0,2].plot(epochs, train_GAN, label="Train GAN", marker='o', markersize=3)
    if len(val_GAN) > 0 and len(val_epochs) > 0:
        axes[0,2].plot(val_epochs, val_GAN, label="Val GAN", marker='x', markersize=5)
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("GAN Loss")
    axes[0,2].set_title("Adversarial Loss"); axes[0,2].legend(); axes[0,2].grid(True)
    
    # Cycle Loss
    axes[1,0].plot(epochs, train_cycle, label="Train Cycle", marker='o', markersize=3)
    if len(val_cycle) > 0 and len(val_epochs) > 0:
        axes[1,0].plot(val_epochs, val_cycle, label="Val Cycle", marker='x', markersize=5)
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("Cycle Loss")
    axes[1,0].set_title("Cycle Consistency Loss"); axes[1,0].legend(); axes[1,0].grid(True)
    
    # Identity Loss
    axes[1,1].plot(epochs, train_identity, label="Train Identity", marker='o', markersize=3)
    if len(val_identity) > 0 and len(val_epochs) > 0:
        axes[1,1].plot(val_epochs, val_identity, label="Val Identity", marker='x', markersize=5)
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Identity Loss")
    axes[1,1].set_title("Identity Loss"); axes[1,1].legend(); axes[1,1].grid(True)
    
    # Combined Loss Comparison
    axes[1,2].plot(epochs, train_G, label="Generator", marker='o', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_D, label="Discriminator", marker='s', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_GAN, label="GAN", marker='^', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_cycle, label="Cycle", marker='v', markersize=2, alpha=0.7)
    axes[1,2].plot(epochs, train_identity, label="Identity", marker='d', markersize=2, alpha=0.7)
    axes[1,2].set_xlabel("Epoch"); axes[1,2].set_ylabel("Loss Value")
    axes[1,2].set_title("All Training Losses"); axes[1,2].legend(); axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"images/graphs/final_all_losses_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual plots as well
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_G, label="Train Generator", marker='o')
    if len(val_G) > 0 and len(val_epochs) > 0:
        plt.plot(val_epochs, val_G, label="Val Generator", marker='x')
    plt.xlabel("Epoch"); plt.ylabel("Generator Loss"); plt.title("Generator Loss vs Epoch")
    plt.legend(); plt.grid(True)
    plt.savefig(f"images/graphs/final_generator_loss_vs_epoch.png")
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_D, label="Train Discriminator", marker='o')
    if len(val_D) > 0 and len(val_epochs) > 0:
        plt.plot(val_epochs, val_D, label="Val Discriminator", marker='x')
    plt.xlabel("Epoch"); plt.ylabel("Discriminator Loss"); plt.title("Discriminator Loss vs Epoch")
    plt.legend(); plt.grid(True)
    plt.savefig(f"images/graphs/final_discriminator_loss_vs_epoch.png")
    plt.close()


# -----------------------
# Original plotting function (kept for compatibility)
# -----------------------
def plot_losses(train_G, val_G, train_D, val_D, train_GAN, val_GAN, train_cycle, val_cycle, 
                train_identity, val_identity, start_epoch, n_epochs, dataset_name):
    epochs = range(start_epoch, n_epochs)
    val_epochs = list(range(start_epoch, n_epochs, 5))  # validation every 5 epochs
    
    # Ensure we have validation data
    if len(val_epochs) > len(val_G):
        val_epochs = val_epochs[:len(val_G)]
    
    # Create subplots for all losses
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CycleGAN Training Losses', fontsize=16)
    
    # Generator Loss
    axes[0,0].plot(epochs, train_G, label="Train Generator", marker='o', markersize=3)
    axes[0,0].plot(val_epochs, val_G, label="Val Generator", marker='x', markersize=5)
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Generator Loss")
    axes[0,0].set_title("Total Generator Loss"); axes[0,0].legend(); axes[0,0].grid(True)
    
    # Discriminator Loss
    axes[0,1].plot(epochs, train_D, label="Train Discriminator", marker='o', markersize=3)
    axes[0,1].plot(val_epochs, val_D, label="Val Discriminator", marker='x', markersize=5)
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Discriminator Loss")
    axes[0,1].set_title("Discriminator Loss"); axes[0,1].legend(); axes[0,1].grid(True)
    
    # GAN/Adversarial Loss
    axes[0,2].plot(epochs, train_GAN, label="Train GAN", marker='o', markersize=3)
    axes[0,2].plot(val_epochs, val_GAN, label="Val GAN", marker='x', markersize=5)
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("GAN Loss")
    axes[0,2].set_title("Adversarial Loss"); axes[0,2].legend(); axes[0,2].grid(True)
    
    # Cycle Loss
    axes[1,0].plot(epochs, train_cycle, label="Train Cycle", marker='o', markersize=3)
    axes[1,0].plot(val_epochs, val_cycle, label="Val Cycle", marker='x', markersize=5)
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("Cycle Loss")
    axes[1,0].set_title("Cycle Consistency Loss"); axes[1,0].legend(); axes[1,0].grid(True)
    
    # Identity Loss
    axes[1,1].plot(epochs, train_identity, label="Train Identity", marker='o', markersize=3)
    axes[1,1].plot(val_epochs, val_identity, label="Val Identity", marker='x', markersize=5)
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Identity Loss")
    axes[1,1].set_title("Identity Loss"); axes[1,1].legend(); axes[1,1].grid(True)
    
    # Combined Loss Comparison
    axes[1,2].plot(epochs, train_G, label="Generator", marker='o', markersize=2)
    axes[1,2].plot(epochs, train_D, label="Discriminator", marker='s', markersize=2)
    axes[1,2].plot(epochs, train_GAN, label="GAN", marker='^', markersize=2)
    axes[1,2].plot(epochs, train_cycle, label="Cycle", marker='v', markersize=2)
    axes[1,2].plot(epochs, train_identity, label="Identity", marker='d', markersize=2)
    axes[1,2].set_xlabel("Epoch"); axes[1,2].set_ylabel("Loss Value")
    axes[1,2].set_title("All Training Losses"); axes[1,2].legend(); axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"images/graphs/all_losses_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual plots as well
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_G, label="Train Generator", marker='o')
    if len(val_G) > 0 and len(val_epochs) > 0:
        plt.plot(val_epochs, val_G, label="Val Generator", marker='x')
    plt.xlabel("Epoch"); plt.ylabel("Generator Loss"); plt.title("Generator Loss vs Epoch")
    plt.legend(); plt.grid(True)
    plt.savefig(f"images/graphs/generator_loss_vs_epoch.png")
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_D, label="Train Discriminator", marker='o')
    if len(val_D) > 0 and len(val_epochs) > 0:
        plt.plot(val_epochs, val_D, label="Val Discriminator", marker='x')
    plt.xlabel("Epoch"); plt.ylabel("Discriminator Loss"); plt.title("Discriminator Loss vs Epoch")
    plt.legend(); plt.grid(True)
    plt.savefig(f"images/graphs/discriminator_loss_vs_epoch.png")
    plt.close()


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows + num_workers>0
    main()

