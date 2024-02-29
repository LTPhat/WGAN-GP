import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import os
from PIL import Image



class FaceDataset(Dataset):
  def __init__(self, path, lim, size=64):
      self.sizes=[size, size]
      items, labels=[],[]

      for data in os.listdir(path)[:lim]:
        #path: './data/celeba/img_align_celeba'
        #data: '114568.jpg
          item = os.path.join(path,data)
          items.append(item)
          labels.append(data)
      self.items=items
      self.labels=labels


  def __len__(self):
      return len(self.items)


  def __getitem__(self,idx):
      data = Image.open(self.items[idx]).convert('RGB') 
      data = np.asarray(torchvision.transforms.Resize(self.sizes)(data)) # 128 x 128 x 3
      data = np.transpose(data, (2,0,1)).astype(np.float32, copy=False) # 3 x 128 x 128 # from 0 to 255
      data = torch.from_numpy(data).div(255) # from 0 to 1
      return data, self.labels[idx]

