import numpy as np
import pandas as pd

import glob
import os
import re
import time
import pickle as pkl
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool
from skimage.metrics import structural_similarity
from sklearn.linear_model import LinearRegression


def metrics_ssim(img1, img2, clip=False):
    data_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    if clip:
        img1 = np.clip(img1, 0., 1.)
        img2 = np.clip(img2, 0., 1.)
    (score, diff) = structural_similarity(img1, img2, full=True, data_range=data_range)
    return score

def single_vol(payload, max_project):
    idx, img1_vol, img2_vol = payload

    assert len(img1_vol) == len(img2_vol)

    if not max_project:
        vol_metric_list = []
        for (img1_slice, img2_slice) in zip(img1_vol, img2_vol):
            curr_metric = metrics_ssim(img1_slice, img2_slice)
            vol_metric_list.append(curr_metric)
        return (idx, np.mean(vol_metric_list))
    else:
        img1, img2 = np.max(img1_vol, axis=0), np.max(img2_vol, axis=0)
        metric = metrics_ssim(img1, img2)
        return (idx, metric)

def ssim_multiprocess(data_dict, k1, k2, max_project):
    metric_list = []

    all_rows = list(zip(np.arange(len(data_dict[k1])), data_dict[k1], data_dict[k2]))
    assert 'gt' not in k1, 'While building linear regression, this changes gt to prediction, try setting k2=gt'
    print(f'Running metrics on {len(all_rows)} test images')
    partial_obj = partial(single_vol, max_project=max_project)

    # parallel
    pool_obj = Pool(20)
    for return_metric in pool_obj.imap_unordered(partial_obj, all_rows):
        metric_list.append(return_metric)
    pool_obj.close()

    return np.array(metric_list)

def main(train_data, split_fpth, split_name, n2v_pred, n2v_N_diff_pred, save_stats_fpth, max_project):
    whole_X, whole_Y = np.load(train_data)['X'].squeeze(axis=1), np.load(train_data)['Y'].squeeze(axis=1)

    with open(split_fpth, 'rb') as pkl_f:
        splits = pkl.load(pkl_f)

    split_X, split_Y = whole_X[splits[split_name]], whole_Y[splits[split_name]],
    n2v = np.load(n2v_pred)['pred']
    n2v_diff = np.load(n2v_N_diff_pred)['pred']

    # To remove
    split_X = split_X[0: len(n2v_diff)]
    split_Y = split_Y[0: len(n2v_diff)]
    n2v = n2v[0: len(n2v_diff)]

    data_dict = {'noise': split_X,
                 'gt': split_Y,
                 'n2v': n2v,
                 'n2v_diff': n2v_diff}

    print(f'Shape of noisy, gt, n2v and n2v + diff: {split_X.shape, split_Y.shape, n2v.shape, n2v_diff.shape}')

    st = time.time()
    noisy_vs_gt_ssim = ssim_multiprocess(data_dict, k1='noise', k2='gt', max_project=max_project)
    n2v_vs_gt_ssim = ssim_multiprocess(data_dict, k1='n2v', k2='gt', max_project=max_project)
    diff_vs_gt_ssim = ssim_multiprocess(data_dict, k1='n2v_diff', k2='gt', max_project=max_project)

    print(f"SSIM condition <-> GT: {noisy_vs_gt_ssim[:, 1].mean()}")
    print(f"SSIM n2v <-> GT: {n2v_vs_gt_ssim[:, 1].mean()}")
    print(f"SSIM +diff <-> GT: {diff_vs_gt_ssim[:, 1].mean()}")
    np.savez_compressed(save_stats_fpth, stats={
        'noisy_vs_gt_ssim': noisy_vs_gt_ssim,
        'n2v_vs_gt_ssim': n2v_vs_gt_ssim,
        'diff_vs_gt_ssim': diff_vs_gt_ssim,
    })
    print(f'Finished in {time.time() - st:.4f} seconds')


if __name__ == '__main__':
    train_data = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/train_data/data_label.npz'
    split_fpth = '/home/sivark/supporting_files/denoising/trained_models/planaria_Train2TrainValTestv2.pkl'
    n2v_pred = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/n2v_pred_Train2TrainValTest_val_idxv2.npz'
    n2v_N_diff_pred = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/Diff_pred_Train2TrainValTest_val_idx_t100.npz'
    save_stats_fpth = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/t100_ssim.npz'

    main(
        train_data=train_data,
        split_fpth=split_fpth,
        split_name='val_idx',
        n2v_pred=n2v_pred,
        n2v_N_diff_pred=n2v_N_diff_pred,
        save_stats_fpth=save_stats_fpth,
        max_project=True,
    )