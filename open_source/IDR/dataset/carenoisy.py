import numpy as np
import os
import pickle
from torch.utils.data import Dataset

from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class BaseDatasetCARENoisy(Dataset):
    def __init__(self, name: str, **kwargs):
        super().__init__()

        print(f'dataset: {name}')

        if name == 'planaria_2D':
            self.npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Planaria/data_label.npz'
            data = np.load(self.npz_path)['X']
            data = self.split_train_test(data)
            self.images = np.transpose(data, (0, 2, 1, 3, 4))
            self.images = np.reshape(self.images, (-1, self.images.shape[2], self.images.shape[3], self.images.shape[4]))
            # self.images = np.transpose(self.images, (0, 2, 3, 1))
            # self.images = np.repeat(self.images, 3, axis=1)
        elif name == 'tribolium_2D':
            self.npz_path = '/home/schaudhary/siva_projects/denoising/data/Denoising_Tribolium/train_data/data_label.npz'
            data = np.load(self.npz_path)['X']
            data = self.split_train_test(data)
            self.images = np.transpose(data, (0, 2, 1, 3, 4))
            self.images = np.reshape(self.images, (-1, self.images.shape[2], self.images.shape[3], self.images.shape[4]))
            # self.images = np.transpose(self.images, (0, 2, 3, 1))
            # self.images = np.repeat(self.images, 3, axis=1)
        elif name == 'flywing':
            self.npz_path = '/home/schaudhary/siva_projects/denoising/data/Projection_Flywing/train_data/data_label.npz'
            data = np.load(self.npz_path)['X']
            data = self.split_train_test(data)
            self.images = data
            # self.images = np.transpose(data, (0, 2, 3, 1))
            # self.images = np.repeat(self.images, 3, axis=1)
        else:
            raise ValueError(f'Unsupported data name {name}')
        
        # print(f'Shape of the dataset before: {self.images.shape}')

        # reshape to n x h x w x c (necessary dimensions for IDR)
        n, c, h, w = self.images.shape
        self.images = np.reshape(self.images, (n, h, w, c))

        print(f'Shape of the dataset: {self.images.shape}')

    def split_train_test(self, data):
        split_train_test_path = os.path.join(os.path.dirname(self.npz_path), 'Train2TrainValTestv2.pkl')
        if os.path.exists(split_train_test_path):
            with open(split_train_test_path, 'rb') as f:
                train_test_idx = pickle.load(f)
            data = data[train_test_idx['train_idx']]
            print('train split used')
        return data

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        im = self.images[index]
        # im = self.transform(im)

        # scale values to be between 0,1
        min = im.min()
        max = im.max()
        im = (im-min) / (max-min)

        return {'gt':im}
