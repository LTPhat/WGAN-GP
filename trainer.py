from params import Parameters
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

params = Parameters()


class Trainer():
    def __init__(self, dataloader, generator, critic, n_epochs, root_path, device = "cuda"):
        super(Trainer, self).__init__()
        self.generator = generator
        self.critic = critic
        self.n_epochs = n_epochs
        self.root_path = root_path
        self.dataloader = dataloader
        self.device = device
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5,0.9))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.5,0.9))

    def gradient_penalty(self, real_img, fake_img, critic, alpha, lamda= 10):
        mix_img = alpha * real_img + (1- alpha) * fake_img # bs x 3 x 128 x 128
        mix_score = critic(mix_img)   # bs x 1 x 1

        gradients = torch.autograd.grad(
            inputs = mix_img,
            outputs = mix_score,
            grad_outputs = torch.ones_like(mix_score),
            retain_graph = True,
            create_graph = True,
            only_inputs = True
        )[0]    # bs x 3 x 128 x 128

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = lamda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train(self):
        total_critic_loss = []
        total_generator_loss = []
        for epoch in range(self.n_epochs):
            for real, _ in tqdm(self.dataloader):
                curr_batch_size = len(real)
                real_img = real.to(self.device)
                curr_step = 0
                # Critic : Gen =  cycle : 1
                mean_critic_loss = 0
                for time in range(params.crit_circle):
                    #---------Train Critic ----------------
                    self.critic_opt.zero_grad()

                    noise = self.generator.gen_noise(height = curr_batch_size, width = params.z_dim)
                    # Get fake img
                    fake_img = self.generator(noise)

                    # Get prediction of critic
                    critic_fake_pred = self.critic(fake_img)
                    critic_real_pred = self.critic(real_img)

                    alpha = torch.rand(len(real),1,1,1,device=self.devicedevice, requires_grad=True)   # bs x 1 x 1 x 1
                    grad_penalty = self.gradient_penalty(real_img = critic_real_pred, fake_img = critic_fake_pred, critic = self.critic, alpha = alpha)

                    # Critic loss function
                    critic_loss =  critic_fake_pred.mean() - critic_real_pred.mean() + grad_penalty

                    mean_critic_loss += critic_loss.item()
                    # Backprob
                    critic_loss.backward()
                    self.critic_opt.step()

                mean_critic_loss = mean_critic_loss / params.crit_circle
                # Add loss per batch to total loss array
                total_critic_loss.append(mean_critic_loss)


                # ------------- Train Generator ------------------

                self.generator_opt.zero_grad()
                noise = self.generator.gen_noise(height = curr_batch_size, width = params.z_dim)

                # Generator fake prediction
                generator_fake_pred = self.generator(noise)

                # Critic fake prediction

                critic_fake_pred = self.critic(generator_fake_pred)

                generator_loss = - critic_fake_pred.mean()

                # Backward

                generator_loss.backward()
                self.generator_opt.step()

                total_generator_loss.append(generator_loss)

                # ---------------Saving and visualizing-------------------
                if curr_step % params.show_step == 0 and curr_step > 0:
                  # Save critic cp
                  self.save_checkpoint(self.root_path, "critic", self.critic, self.critic_opt, epoch, mean_critic_loss)
                  # Save generator cp
                  self.save_checkpoint(self.root_path, "generator", self.generator, self.generator_opt, epoch, generator_loss)

                
                if curr_step % params.show_step == 0:
                    # show real image
                    self.show_samples(real_img, "real")
                    # show fake image
                    self.show_samples(generator_fake_pred, "fake")
                
                    print(f"Epoch: {epoch}: Step {curr_step}: Generator loss: {generator_loss}, critic loss: {mean_critic_loss}")

                    plt.plot(
                        range(len(total_generator_loss)),
                        torch.Tensor(total_generator_loss),
                        label="Generator Loss"
                    )

                    plt.plot(
                        range(len(total_critic_loss)),
                        torch.Tensor(total_critic_loss),
                        label="Critic Loss"
                    )

                    plt.ylim(-150,150)
                    plt.legend()
                    plt.show()
                  
                
                curr_step += 1


    def save_checkpoint(self, root_path, name, model, optimizer, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, root_path + "/best_{}.pkl".format(name))
        print("Successfully saved checkpoint at epoch {} with loss {}".format(epoch, loss))


    def load_checkpoint(self, root_path, name, model, optimizer):
        checkpoint = torch.load(root_path + "/best_{}.pkl".format(name))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded checkpoint")



    def show_samples(self, data, title, n_samples = 16, n_rows = 4):
        data = data.detach().cpu()
        grid = make_grid(data[:n_samples], nrow=4).permute(1,2,0)
        plt.imshow(grid.clip(0,1))
        plt.title(title)
        plt.show()


