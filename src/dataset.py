from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from torchvision.io import read_image
from pickle import load as read_pickle, dump as write_pickle

class CifarDataset(Dataset):
  def __init__(self, train:bool = True):
    self.path = Path('data', 'cifar-10', 'cifar-10-python', 'cifar-10-batches-py')
    if train:
      batches = []
      for i in range(5):
        with (self.path / f'data_batch_{i + 1}').open('rb') as file:
          batches.append(read_pickle(file, encoding='bytes'))
    else:
      with (self.path / 'test_batch').open('rb') as file:
        batches = [read_pickle(file, encoding='bytes')]
    self.images, self.labels = list(zip(
      *[(batch[b'data'].reshape((10000,3,32,32)), batch[b'labels']) for batch in batches]
    ))
    self.images = np.concat(self.images)
    self.labels = np.concat(self.labels)
    # Binary label for Cat (class 3)
    self.labels = (self.labels == 3).astype(int)
    self.label_names = {1: "cat", 0: "other"}

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    image = self.images[index]
    label = self.labels[index]
    return image, label