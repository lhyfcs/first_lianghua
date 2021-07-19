from torch.utils import data
import torch
import numpy as np
import os
from torchvision import transforms as T

class SocketTrainDataLoader(data.Dataset):
    def __init__(self, root, train = True):
        self.train = train
        if train:
            self.dataSet = np.load(file=os.path.join(root, 'train.npy'), allow_pickle=True)
        else:
            self.dataSet = np.load(file=os.path.join(root, 'test.npy'), allow_pickle=True)
        if train:
            self.target = np.loadtxt(fname=os.path.join(root, 'target_train.csv'), dtype=np.int, delimiter=',')
        else:
            self.target = np.loadtxt(fname=os.path.join(root, 'target_test.csv'), dtype=np.int, delimiter=',')
        self.transforms = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.dataSet)

    def __getitem__(self, index):
        data = self.dataSet[index]
        # data = self.transforms(data)
        # targetTensor = torch.zeros(200)
        target = torch.tensor(self.target[index] + 100)
        # targetTensor[target] = 1
        return data, target