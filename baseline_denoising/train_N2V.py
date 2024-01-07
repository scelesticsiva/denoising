from n2v.models import N2VConfig, N2V
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
    X = X[:10000]
    Y = Y[:10000]
    train_X, train_Y, val_X, val_Y = split_data(X, Y, 0.2)

    train_X = np.reshape(train_X, (-1, train_X.shape[2], train_X.shape[3], train_X.shape[4]))
    val_X = np.reshape(val_X, (-1, val_X.shape[2], val_X.shape[3], val_X.shape[4]))

    config = N2VConfig(train_X, unet_kern_size=3,
                   train_steps_per_epoch=400, train_epochs=200, train_loss='mse', batch_norm=True, 
                   train_batch_size=16, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64), 
                   unet_n_first = 32,
                   unet_residual = True,
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=2,
                   single_net_per_channel=False)

    # Let's look at the parameters stored in the config-object.
    print(vars(config))

    model = N2V(config, 'N2V_planaria', basedir='/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/baseline_denoising/N2V')

    history = model.train(train_X, val_X)