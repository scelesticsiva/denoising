import numpy as np
import pandas as pd

import json
import glob
import os
import re
import time
import pickle as pkl
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool
from skimage.metrics import structural_similarity


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
    #print(f'Running metrics on {len(all_rows)} test images')
    partial_obj = partial(single_vol, max_project=max_project)

    # parallel
    pool_obj = Pool(20)
    for return_metric in pool_obj.imap_unordered(partial_obj, all_rows):
        metric_list.append(return_metric)
    pool_obj.close()

    return np.array(metric_list)

def main(
    data_npz,
    tune_dir,
    save_stats_fpth, 
    max_project):

    st = time.time()

    # load everything except diff results from data_npz
    data_dict = {
	'noise': np.load(data_npz)['noisy'],
	'gt': np.load(data_npz)['clean'],
	'n2v': np.load(data_npz)['n2v_only'],
    }

    noisy_vs_gt_ssim = ssim_multiprocess(data_dict, k1='noise', k2='gt', max_project=max_project)
    n2v_vs_gt_ssim = ssim_multiprocess(data_dict, k1='n2v', k2='gt', max_project=max_project)
    stats = {
        'noisy_vs_gt_ssim': noisy_vs_gt_ssim,
        'n2v_vs_gt_ssim': n2v_vs_gt_ssim,
    }
    for k, v in stats.items():
        print(k, f"{v[:, 1].mean():.03f}")

    # load diff results from the tuning dir
    tune_run_npz_list = glob.glob(os.path.join(tune_dir, "run*.npz"))
    for run_npz in tune_run_npz_list:
        #print(run_npz)
        run_json = run_npz.replace('.npz', '.json')
        with open(run_json) as f:
            run_params = json.load(f)
        del run_params['save_fpth']
        run_id = os.path.basename(run_npz).replace('.npz', '')
        data_dict[run_id] = np.load(run_npz)['pred']
        diff_vs_gt_ssim = ssim_multiprocess(data_dict, k1=run_id, k2='gt', max_project=max_project)
        stats[f'diff_{run_id}_vs_gt_ssim'] = diff_vs_gt_ssim
        print(run_params, f"{diff_vs_gt_ssim[:,1].mean():.03f}")
    
    np.savez_compressed(save_stats_fpth, stats=stats)
    print(f'Finished in {time.time() - st:.4f} seconds')


if __name__ == '__main__':
    #data_npz = '/home/mds/data/denoising/datasets/clean_noisy_n2v_val_set.npz'
    #tune_dir = '/home/mds/data/denoising/hparam_tuning/diffusion_outputs/val_all/v1'
    data_npz = '/home/mds/data/denoising/datasets/clean_noisy_n2v_val_set_subset10.npz'
    tune_dir = '/home/mds/data/denoising/hparam_tuning/diffusion_outputs/val_subset10/v9'

    save_stats_fpth = os.path.join(tune_dir, 'ssim_results.npz')

    main(
        data_npz=data_npz,
        tune_dir=tune_dir,
        save_stats_fpth=save_stats_fpth,
        max_project=True,
    )
