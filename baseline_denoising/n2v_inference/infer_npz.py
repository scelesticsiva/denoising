import json
import glob
import os
import tifffile
import numpy as np
import pickle as pkl

from munch import Munch
from PIL import Image
from n2v.models import N2VConfig, N2V

from functools import partial
from multiprocessing import Pool

def predict_single_vol(single_vol, model):
    curr_vol = np.expand_dims(single_vol, axis=-1)
    all_preds = []
    for curr_slice in curr_vol:
        curr_pred = model.predict(curr_slice, axes='YXC')
        all_preds.append(curr_pred[..., 0][None, ...])

    return np.concatenate(all_preds, axis=0)

def read_tif_as_npy(tif_fpth):
    img = tifffile.imread(tif_fpth).astype(np.float32)
    return img

def infer(config):
    model = N2V(config=None, name=config.model_name, basedir=config.basedir)
    model.load_weights(name=config.weights_name)

    whole_dataset = np.load(config.data_fpth)['X'].squeeze(axis=1)

    with open(config.split_fpth, 'rb') as pkl_f:
        splits = pkl.load(pkl_f)

    subset = whole_dataset[splits[config.split_to_infer]]
    print(f'Total number of noisy images found: {len(subset)}')

    results = []
    partial_obj = partial(predict_single_vol, model=model)

    for single_vol in subset:
        results.append(partial_obj(single_vol))

    results_save_fpth = config.pred_save_fpth.replace('.npz', f'_{config.split_to_infer}.npz')
    np.savez_compressed(results_save_fpth, pred=np.concatenate(results, axis=0))


if __name__ == '__main__':
    infer_config = Munch.fromDict(dict(
        model_name='N2V_planaria_2D_2',
        basedir='/home/sivark/supporting_files/denoising/trained_models/planaria_Train2TrainValTest/',
        data_fpth=f'/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/train_data/data_label.npz',
        weights_name='weights_best.h5',
        split_fpth='/home/sivark/supporting_files/denoising/trained_models/planaria_Train2TrainValTestv2.pkl',
        pred_save_fpth=f'/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/n2v_pred_Train2TrainValTest.npz',
        split_to_infer='val_idx',
    ))

    infer(infer_config)