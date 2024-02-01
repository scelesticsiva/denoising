import numpy as np
import pandas as pd

import time
import pickle as pkl
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool
from skimage.metrics import structural_similarity
from sklearn.linear_model import LinearRegression
import lpips
import torch
from pytorch_fid_master.src.pytorch_fid.fid_score import calculate_frechet_distance, calculate_activation_statistics
from pytorch_fid_master.src.pytorch_fid.inception import InceptionV3

def metrics_ssim(img1, img2, clip=False):
    data_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    if clip:
        img1 = np.clip(img1, 0., 1.)
        img2 = np.clip(img2, 0., 1.)
    (score, diff) = structural_similarity(img1, img2, full=True, data_range=data_range)
    return score

def metrics_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    if max_value == 1:
        img1 = (img1 - np.min(img1))/(np.max(img1) - np.min(img1))
        img1 = (img2 - np.min(img2))/(np.max(img2) - np.min(img2))
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def metrics_lpips(img1, img2): 
    img1 = torch.tensor(img1, dtype=torch.float32)
    img2 = torch.tensor(img2, dtype=torch.float32)
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    score = loss_fn_alex(img1, img2)
    return score.item()

def metrics_fid(img1, img2, dims=2048, num_workers=0, batch_size=8):
    print(f'Calculating FID using {dims} dims')

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).float()
    m1, s1 = calculate_activation_statistics([img1], model, batch_size=batch_size, dims=dims,
                                    device=device, num_workers=num_workers
                                    )
    m2, s2 = calculate_activation_statistics([img2], model, batch_size=batch_size, dims=dims,
                                    device=device, num_workers=num_workers
                                    )
    score = calculate_frechet_distance(m1, s1, m2, s2)
    return score

def single_vol(payload, max_project, metric, fid_dims=None):
    idx, img1_vol, img2_vol = payload

    assert len(img1_vol) == len(img2_vol)

    if not max_project:
        vol_metric_list = []
        for (img1_slice, img2_slice) in zip(img1_vol, img2_vol):
            if metric == 'ssim':
                curr_metric = metrics_ssim(img1_slice, img2_slice)
            elif metric == 'psnr':
                curr_metric = metrics_psnr(img1_slice, img2_slice)
            elif metric == 'lpips':
                curr_metric = metrics_lpips(img1_slice, img2_slice)
            elif metric == 'fid':
                curr_metric = metrics_fid(img1_slice, img2_slice, fid_dims)
            vol_metric_list.append(curr_metric)
        return (idx, np.mean(vol_metric_list))
    else:
        img1, img2 = np.max(img1_vol, axis=0), np.max(img2_vol, axis=0)
        if metric == 'ssim':
            single_metric = metrics_ssim(img1, img2)
        elif metric == 'psnr':
            single_metric = metrics_psnr(img1, img2)
        elif metric == 'lpips':
            single_metric = metrics_lpips(img1, img2)
        elif metric == 'fid':
            single_metric = metrics_fid(img1, img2, fid_dims)
        return (idx, single_metric)

def metric_multiprocess(data_dict, k1, k2, max_project, metric, fid_dims=None):
    metric_list = []

    all_rows = list(zip(np.arange(len(data_dict[k1])), data_dict[k1], data_dict[k2]))
    assert 'gt' not in k1, 'While building linear regression, this changes gt to prediction, try setting k2=gt'
    print(f'Running metrics on {len(all_rows)} test images')
    partial_obj = partial(single_vol, max_project=max_project, metric=metric, fid_dims=fid_dims)

    # parallel
    pool_obj = Pool(20)
    for return_metric in pool_obj.imap_unordered(partial_obj, all_rows):
        metric_list.append(return_metric)
    pool_obj.close()

    return np.array(metric_list)

def main(train_data, split_fpth, split_name, n2v_pred, n2v_N_diff_pred, save_stats_fpth, metrics, max_project, fid_dims=2048):
    whole_X, whole_Y = np.load(train_data)['X'].squeeze(axis=1), np.load(train_data)['Y'].squeeze(axis=1)
    with open(split_fpth, 'rb') as pkl_f:
        splits = pkl.load(pkl_f)

    split_X, split_Y = whole_X[splits[split_name]], whole_Y[splits[split_name]],
    n2v = np.load(n2v_pred)['pred']
    n2v_diff = np.load(n2v_N_diff_pred)['pred']#[:8]

    # To remove
    split_X = split_X[0: len(n2v_diff)]
    split_Y = split_Y[0: len(n2v_diff)]
    n2v = n2v[0: len(n2v_diff)]

    data_dict = {'noise': split_X,
                 'gt': split_Y,
                 'n2v': n2v,
                 'n2v_diff': n2v_diff}

    print(f'Shape of noisy, gt, n2v and n2v + diff: {split_X.shape, split_Y.shape, n2v.shape, n2v_diff.shape}')

    for metric in metrics:
        st = time.time()

        noisy_vs_gt_score = metric_multiprocess(data_dict, k1='noise', k2='gt', max_project=max_project, metric=metric, fid_dims=fid_dims)
        n2v_vs_gt_score = metric_multiprocess(data_dict, k1='n2v', k2='gt', max_project=max_project, metric=metric, fid_dims=fid_dims)
        diff_vs_gt_score = metric_multiprocess(data_dict, k1='n2v_diff', k2='gt', max_project=max_project, metric=metric, fid_dims=fid_dims)

        print(f"{metric.upper()} condition <-> GT: {noisy_vs_gt_score[:, 1].mean()}")
        print(f"{metric.upper()} n2v <-> GT: {n2v_vs_gt_score[:, 1].mean()}")
        print(f"{metric.upper()} +diff <-> GT: {diff_vs_gt_score[:, 1].mean()}")

        save_fpth = save_stats_fpth.replace('.npz', f'_{metric}.npz')

        np.savez_compressed(save_fpth, stats={
            'noisy_vs_gt_score': noisy_vs_gt_score,
            'n2v_vs_gt_score': n2v_vs_gt_score,
            'diff_vs_gt_score': diff_vs_gt_score,
        })
        print(f'Finished {metric.upper()} in {time.time() - st:.4f} seconds')


if __name__ == '__main__':
    train_data = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/train_data/data_label.npz'
    split_fpth = '/home/sivark/supporting_files/denoising/trained_models/planaria_Train2TrainValTestv2.pkl'
    n2v_pred = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/n2v_pred_Train2TrainValTest_val_idxv2.npz'
    n2v_N_diff_pred = '/home/joy/project_repos/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/Diff_mixing_clean_pred_Train2TrainValTest_val_idx_t100_ns1.npz'
    save_stats_fpth = '/home/joy/project_repos/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/mixing_clean_subset_ssim.npz'

    main(
        train_data=train_data,
        split_fpth=split_fpth,
        split_name='val_idx',
        n2v_pred=n2v_pred,
        n2v_N_diff_pred=n2v_N_diff_pred,
        save_stats_fpth=save_stats_fpth,
        metrics=['ssim', 'psnr', 'fid'],
        max_project=True,
        fid_dims=768 # 2048 (final average pooling); 768 (pre-aux classifier); 192 (second max pooling); 64 (first max pooling)
    )