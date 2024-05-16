import numpy as np
import os
import pickle
import torchvision as thv
from torch.utils.data import Dataset

class BaseDatasetCARENoisy(Dataset):
    def __init__(self,
                 name: str, 
                 split: str):
        super().__init__()
        self.name = name
        self.split = split

        if name == 'planaria_2D':
            self.npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Planaria/data_label.npz'
            data = np.load(self.npz_path)['X']
            data = self.split_train_test(data)
            self.images = np.transpose(data, (0, 2, 1, 3, 4))
            self.images = np.reshape(self.images, (-1, self.images.shape[2], self.images.shape[3], self.images.shape[4]))
            # self.images = np.transpose(self.images, (0, 2, 3, 1))
            self.images = np.repeat(self.images, 3, axis=1)
        elif name == 'tribolium_2D':
            self.npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Tribolium/train_data/data_label.npz'
            data = np.load(self.npz_path)['X']
            data = self.split_train_test(data)
            self.images = np.transpose(data, (0, 2, 1, 3, 4))
            self.images = np.reshape(self.images, (-1, self.images.shape[2], self.images.shape[3], self.images.shape[4]))
            # self.images = np.transpose(self.images, (0, 2, 3, 1))
            self.images = np.repeat(self.images, 3, axis=1)
        elif name == 'flywing':
            self.npz_path = '/home/schaudhary/siva_projects/denoising/data/Projection_Flywing/train_data/data_label.npz'
            data = np.load(self.npz_path)['X']
            data = self.split_train_test(data)
            self.images = np.transpose(data, (0, 2, 1, 3, 4))
            self.images = np.reshape(self.images, (-1, self.images.shape[2], self.images.shape[3], self.images.shape[4]))
            # self.images = np.transpose(data, (0, 2, 3, 1))
            self.images = np.repeat(self.images, 3, axis=1)
        else:
            raise ValueError(f'Unsupported data name {name}')
        
        print(f'Shape of the dataset: {self.images.shape}')

    def split_train_test(self, data):
        split_train_test_path = os.path.join(os.path.dirname(self.npz_path), 'Train2TrainValTestv2.pkl')
        if os.path.exists(split_train_test_path):
            with open(split_train_test_path, 'rb') as f:
                train_test_idx = pickle.load(f)
            data = data[train_test_idx[self.split]]
            print(f'{self.split} split used')
        return data

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        im = self.images[index]
        # im = self.transform(im)
        return {'L':im}