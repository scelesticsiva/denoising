import glob
import numpy as np

from torch.utils import data


class NPZ3DDataset(data.Dataset):
    def __init__(self, all_files, key):
        self.files = all_files
        self.key = key
        print(f'Number of files found: {len(self.files)}')

    @classmethod
    def from_dir(cls, dir, **kwargs):
        all_files = glob.glob(dir + '*.npz')
        return cls(all_files, **kwargs)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        return {'fpth': file, 'img': np.load(file, allow_pickle=True)[self.key].astype(np.float32)}