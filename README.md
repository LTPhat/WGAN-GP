# WGAN-GP

Implement Wasserstein GAN with Gradient Penalty to synthesize facial images from CelebA Dataset.

## Architecture

![alt text](https://github.com/LTPhat/WGAN-GP/blob/main/res/wgan.jpg)

- Discriminator/Critic loss (with gradient penalty):

![alt text](https://github.com/LTPhat/WGAN-GP/blob/main/res/critic.png)

- Generator loss:

![alt text](https://github.com/LTPhat/WGAN-GP/blob/main/res/generator.png)

## Train WGAN

- **Step 1:** Download CeleA dataset at https://www.kaggle.com/datasets/jessicali9530/celeba-dataset and save at ``./dataset`` folder in this project.
For example: ./dataset/celeba/img_align_celeba.
- **Step 2:** Train WGAN and see results at ``main.py``

```python
python main.py --n_epochs  --batch_size --train_samples --latent_dim --n_critics --show_step
```
```sh
  --n_epochs N_EPOCHS             number of epochs of training
  --batch_size BATCH_SIZE         size of the batches
  --train_samples TRAIN_SAMPLES   number of training images used from dataset
  --latent_dim LATENT_DIM         dimensionality of the latent space
  --n_critic N_CRITIC             number of training steps to train discriminator/critic per iter
  --show_step SHOW_STEP           number of training steps at each epoch to show results
```

## Result
### Loss:
![alt text](https://github.com/LTPhat/WGAN-GP/blob/main/res/loss.png)

### Images:
- Epoch 1:

| Real                                | Fake                                |
| ----------------------------------- | ----------------------------------- |
| ![cat](https://github.com/LTPhat/WGAN-GP/blob/main/res/epoch1_real.png) | ![dog](https://github.com/LTPhat/WGAN-GP/blob/main/res/epoch1_fake.png)|


- Epoch 10:
  
| Real                                | Fake                          |
| ----------------------------------- | ----------------------------------- |
| ![cat](https://github.com/LTPhat/WGAN-GP/blob/main/res/epoch10_real.png) | ![dog](https://github.com/LTPhat/WGAN-GP/blob/main/res/epoch10_fake.png) |

- Epoch 25:
  
| Real                                | Fake                          |
| ----------------------------------- | ----------------------------------- |
| ![cat](https://github.com/LTPhat/WGAN-GP/blob/main/res/epoch25_real.png) | ![dog](https://github.com/LTPhat/WGAN-GP/blob/main/res/epoch25_fake.png) |





