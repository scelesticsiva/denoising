from csbdeep.models import Config, CARE
import numpy as np
import random


def split_data(X, Y, split_ratio=0.2):
    split_ratio = 0.2
    num_examples = X.shape[0]
    idx = [i for i in range(num_examples)]
    random.shuffle(idx)
    validation_examples = idx[:int(num_examples*split_ratio)]
    test_X = X[validation_examples]
    test_Y = Y[validation_examples]
    train_examples = idx[int(num_examples*split_ratio)::]
    train_X = X[train_examples]
    train_Y = Y[train_examples]
    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    planaria_data = np.load('/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/CARE-dataset/Denoising_Planaria/train_data/data_label.npz')
    X = np.transpose(planaria_data['X'], (0, 2, 3, 4, 1))
    Y = np.transpose(planaria_data['Y'], (0, 2, 3, 4, 1))
    # dummy_X = X[0:1000]
    # dummy_Y = Y[0:1000]
    train_X, train_Y, val_X, val_Y = split_data(X, Y, 0.2)

    axes = 'SZYXC'
    n_channel_in = 1
    n_channel_out = 1
    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=10)
    vars(config)

    model = CARE(config, 'care_planaria_3D', basedir='/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/baseline_denoising/care')

    history = model.train(train_X, train_Y, validation_data=(val_X, val_Y))