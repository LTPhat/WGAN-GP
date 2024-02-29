import torch
from torch import nn
import torchvision
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim = 64, base_dim = 16):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim

        def transpose_conv_block(in_channels, out_channels, kernel_size, padding, stride):
            layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, padding = padding, stride = stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace = True)
            )
            return layers

        def gen_noise(height, width = z_dim, device = 'cuda'):
            return torch.rand(height, width)


        self.gen = nn.Sequential(
            # trans conv: dim_out = (dim_in - 1) * stride - 2 * pad + kernel_size
            transpose_conv_block(in_channels= self.z_dim, out_channels= self.base_dim * 32, kernel_size = 4, padding = 0, stride = 1),  # (64, 1, 1) --> (512 ,4, 4)
            transpose_conv_block(in_channels = base_dim * 32, out_channels = base_dim * 16, kernel_size = 4, padding = 1, stride = 2), #(512, 4, 4) --> (256, 8, 8)
            transpose_conv_block(in_channels = base_dim * 16, out_channels= base_dim * 8, kernel_size = 4, padding = 1, stride = 2), # (256, 8, 8) --> (128, 16, 16)
            transpose_conv_block(in_channels = base_dim * 8, out_channels= base_dim * 4, kernel_size = 4, padding = 1, stride = 2),# (128, 16, 16) --> (64, 32, 32)
            transpose_conv_block(in_channels = base_dim * 4, out_channels= base_dim * 2, kernel_size = 4, padding = 1, stride = 2),# (64, 32, 32) --> (32,64,64)
            nn.ConvTranspose2d(in_channels = base_dim * 2, out_channels = 3, kernel_size = 4, padding = 1, stride = 2),# (64, 32, 32) --> (3,128,128)
            nn.Tanh()
        )

        def forward(self, noise):
          z = noise.view(len(noise),self.z_dim , (1, 1))
          x = self.gen(z)
          return x





class Critic(nn.Module):
    def __init__(self, base_dim = 16, img_channels = 3):
        super(Critic, self).__init__()
        self.base_dim = 16
        self.img_channels = 3

        def conv_block(in_channels, out_channels, kernel_size, padding, stride):
            layers = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= kernel_size, padding = padding, stride = stride),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace = True)
            )
            return layers

        self.crit = nn.Sequential(
            conv_block(in_channels = self.img_channels, out_channels = base_dim, kernel_size = 4, padding = 1, stride = 2),  # (3, 128, 128) --> (16, 64, 64)
            conv_block(in_channels = base_dim, out_channels = base_dim * 2, kernel_size = 4, padding = 1, stride = 2), #(16, 64, 64) --> (32, 32, 32)
            conv_block(in_channels = base_dim * 2, out_channels = base_dim * 4, kernel_size = 4, padding = 1,stride = 2),# (32, 32, 32) --> (64, 16, 16)
            conv_block(in_channels = base_dim * 4, out_channels = base_dim * 8, kernel_size = 4, padding = 1,stride = 2),# (64, 16, 16) --> (128, 8, 8)
            conv_block(in_channels = base_dim * 8, out_channels = base_dim * 16, kernel_size = 4, padding = 1,stride = 2),# (128, 8, 8) --> (256, 4, 4)
            nn.Conv2d(in_channels = base_dim * 16, out_channels = 1, kernel_size = 4, padding = 0, stride = 1),  # (256, 4, 4) --> (128, 1, 1)
        )

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
          torch.nn.init.normal_(m.weight, 0.0, 0.02)
          torch.nn.init.constant_(m.bias,0)

        if isinstance(m,nn.BatchNorm2d):
          torch.nn.init.normal_(m.weight, 0.0, 0.02)
          torch.nn.init.constant_(m.bias,0)


    def forward(self, image):
        x = self.crit(image)
        x = x.view(len(x), -1)
        return x


if __name__ == "__main__":
    image = torch.rand((128 ,3, 128, 128))  # Bs 128 images
    G = Generator()
    print(G)
    total_params = sum(p.numel() for p in G.parameters())
    print(f"Total Parameters: {total_params}")


    C = Critic()
    print(C)
    total_params = sum(p.numel() for p in G.parameters())
    print(f"Total Parameters: {total_params}")

    fake = G(image)
    out = C(fake)

    print("Generator output:", fake.shape)
    print("Critic output:", out.shape)

    
