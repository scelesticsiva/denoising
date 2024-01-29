import numpy as np
import pandas as pd

import glob
import os
import re
import time
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool
from skimage.metrics import structural_similarity
from sklearn.linear_model import LinearRegression

class Stitcher:
    def __init__(self, all_files, key, desired_xy=(1024, 1024)):
        self.files = all_files
        self.key = key
        self.n_channel = self.determine_ch_size()
        self.x, self.y = desired_xy
        self.x_pos_pattern = r"x(\d{5})"
        self.y_pos_pattern = r"y(\d{5})"
        self.ps = 64

    def determine_ch_size(self):
        loaded = np.load(self.files[0])[self.key]
        return len(loaded)

    def stitch(self):
        stitched_img = np.zeros((self.n_channel, self.x, self.y))
        for f in self.files:
            fname = os.path.basename(f)
            xpos = int(re.findall(self.x_pos_pattern, fname)[0].split('x')[-1])
            ypos = int(re.findall(self.y_pos_pattern, fname)[0].split('x')[-1])
            stitched_img[:, xpos: xpos + self.ps, ypos: ypos + self.ps] = np.load(f)[self.key]
        return stitched_img

def metrics_ssim(img1, img2, clip=False):
    data_range = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    if clip:
        img1 = np.clip(img1, 0., 1.)
        img2 = np.clip(img2, 0., 1.)
    (score, diff) = structural_similarity(img1, img2, full=True, data_range=data_range)
    return score

def affine_normalize(img1, img2):
    X, Y = img1.reshape(-1, 1), img2.reshape(-1, 1)
    mdl = LinearRegression().fit(X, Y)
    return mdl.predict(X).reshape(img1.shape)

def single_vol(payload, k1, k2, npy_k1, npy_k2):
    img1_fpth, img2_fpth = payload[k1], payload[k2]
    img1_vol = np.load(img1_fpth)[npy_k1]
    img2_vol = np.load(img2_fpth)[npy_k2]
    assert 'gt' not in k1, 'While building linear regression, this changes gt to prediction, try setting k2=gt'
    assert len(img1_vol) == len(img2_vol)
    vol_metric_list = []
    for (img1_slice, img2_slice) in zip(img1_vol, img2_vol):
        img1_slice = affine_normalize(img1_slice, img2_slice)
        curr_metric = metrics_ssim(img1_slice, img2_slice)
        vol_metric_list.append(curr_metric)
    return np.mean(vol_metric_list)

def single_volv2(group_df, k1, k2, npy_k1, npy_k2):
    img1 = Stitcher(group_df[k1].to_numpy(), npy_k1).stitch()
    img2 = Stitcher(group_df[k2].to_numpy(), npy_k2).stitch()

    img1 = np.max(img1, axis=0)
    img2 = np.max(img2, axis=0)

    # fig, axes = plt.subplots(1, 2, figsize=(30, 15))
    # axes[0].imshow(img1, cmap='plasma', vmin=0., vmax=1.)
    # axes[0].set_title(k1)
    # axes[1].imshow(img2, cmap='plasma', vmin=0., vmax=1.)
    # axes[1].set_title(k2)
    # plt.show()

    img1 = affine_normalize(img1, img2)
    curr_metric = metrics_ssim(img1, img2)

    # vol_metric_list = []
    # for (img1_slice, img2_slice) in zip(img1, img2):
    #     curr_metric = metrics_ssim(img1_slice, img2_slice)
    #     vol_metric_list.append(curr_metric)
    return curr_metric

def ssim_multiprocess(input_df, k1, k2, npy_k1, npy_k2):
    metric_list = []

    all_rows = [group_df for _, group_df in input_df.groupby('test_img') if len(group_df) == 256]
    print(f'Running metrics on {len(all_rows)} test images')
    partial_obj = partial(single_volv2, k1=k1, k2=k2, npy_k1=npy_k1, npy_k2=npy_k2)

    # parallel
    pool_obj = Pool(20)
    for return_metric in pool_obj.imap_unordered(partial_obj, all_rows):
        metric_list.append(return_metric)
    pool_obj.close()

    # Serial
    # for idx, each in enumerate(all_rows):
    #     metric_list.append(partial_obj(each))
    #     if idx == 3:
    #         break

    return np.mean(metric_list)

def main(input_df):
    input_df['test_img'] = input_df['fname'].apply(lambda x: re.split(r'p(\d{1,3})', x)[0])  # p is the patch index
    st = time.time()

    noisy_vs_gt_ssim = ssim_multiprocess(input_df, k1='noisy_img', k2='gt_img', npy_k1='img', npy_k2='img')
    n2v_vs_gt_ssim = ssim_multiprocess(input_df, k1='n2v_pred', k2='gt_img', npy_k1='pred', npy_k2='img')
    diff_vs_gt_ssim = ssim_multiprocess(input_df, k1='diff_pred', k2='gt_img', npy_k1='pred', npy_k2='img')

    print(f"SSIM condition <-> GT: {noisy_vs_gt_ssim}")
    print(f"SSIM n2v <-> GT: {n2v_vs_gt_ssim}")
    print(f"SSIM n2v+diff <-> GT: {diff_vs_gt_ssim}")
    print(f'Finished in {time.time() - st:.4f} seconds')


if __name__ == '__main__':
    n2v_pred_dir = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/n2v_pred/condition_1/'
    diff_pred_dir = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/diffusion_pred/t500_from_n2v/condition_1/'
    noisy_img_dir = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/test_data_preprocessed/condition_1/'
    gt_img_dir = '/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/test_data_preprocessed/GT/'

    n2v_pred_npz = {os.path.basename(f): f for f in glob.glob(n2v_pred_dir + '*.npz')}
    diff_pred_npz = {os.path.basename(f): f for f in glob.glob(diff_pred_dir + '*.npz')}
    noisy_img_npz = {os.path.basename(f): f for f in glob.glob(noisy_img_dir + '*.npz')}
    gt_img_npz = {os.path.basename(f): f for f in glob.glob(gt_img_dir + '*.npz')}

    print(len(n2v_pred_npz), len(diff_pred_npz), len(noisy_img_npz), len(gt_img_npz))

    all_pred = list(diff_pred_npz.keys())
    df_dict = dict(zip(['fname', 'diff_pred', 'n2v_pred', 'noisy_img', 'gt_img'],
                       [all_pred, [diff_pred_npz[f] for f in all_pred], [n2v_pred_npz[f] for f in all_pred],
                        [noisy_img_npz[f] for f in all_pred], [gt_img_npz[f] for f in all_pred]]))
    df = pd.DataFrame.from_dict(df_dict)
    main(df)

    # print(len(n2v_pred_npz), len(noisy_img_npz), len(gt_img_npz))
    # all_pred = list(n2v_pred_npz.keys())
    # df_dict = dict(zip(['fname', 'n2v_pred', 'noisy_img', 'gt_img'],
    #                    [all_pred,  [n2v_pred_npz[f] for f in all_pred],
    #                     [noisy_img_npz[f] for f in all_pred], [gt_img_npz[f] for f in all_pred]]))
    # df = pd.DataFrame.from_dict(df_dict)
    # main(df)