from dataset import FaceDataset
from trainer import Trainer
from params import Parameters
from models.network import Critic, Generator
import argparse
import os
import numpy as np
import math
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--train_samples", type=int, default=20000, help="number of training images")
parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--show_step", type=int, default=70, help="number of training steps at each epoch to show result")

opt = parser.parse_args()

# Adjust params 
params = Parameters()
params.batch_size = opt.batch_size
params.epochs = opt.n_epochs
params.z_dim = opt.latent_dim
params.crit_circle = opt.n_critic
params.show_step = opt.show_step
params.train_samples = opt.train_samples

# Dataset


dataset = FaceDataset(path="./dataset", lim=params.train_samples)

dataloader = DataLoader(dataset, batch_size= params.batch_size, shuffle=True)

## Models
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = Generator(params.z_dim).to(device)
critic = Critic().to(device)

if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")

trainer = Trainer(dataloader=dataloader, generator=generator, critic=critic, 
                  n_epochs= params.epochs, device=device, root_path="./checkpoint")

trainer.train()

# Show fake samples after training

noise = torch.rand((params.batch_size, params.z_dim))
fake = trainer.generator(noise)
trainer.show_samples(fake, "fake")
