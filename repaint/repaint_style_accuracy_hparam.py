import argparse
from math import ceil
from omegaconf import OmegaConf
from sampler import BaseSampler
from utils import util_common
from utils import util_image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision as thv
import tifffile
from skimage.metrics import structural_similarity
import pickle
import itertools
import pandas as pd
import sys
from functools import partial
from custom_samplers import DiffusionSampler


def create_diffusion_sampler(data_name):
    if data_name == 'planaria_2D':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_care_planaria_2D.yaml'
    elif data_name == 'tribolium_2D':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_care_tribolium_2D.yaml'
    elif data_name == 'flywing':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_care_flywing.yaml'
    elif data_name == 'synthetic_tubulin_gfp':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_care_synthetic_tubulin_gfp.yaml'
    elif data_name == 'synthetic_tubulin_granules_channel_granules':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_care_synthetic_tubulin_granules_channel_granules.yaml'
    elif data_name == 'synthetic_tubulin_granules_channel_tubules':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_care_synthetic_tubulin_granules_channel_tubules.yaml'
    elif data_name == 'niddl':
        cfg_path = '/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/configs/sample/iddpm_niddl_wb.yaml'
    else:
        raise ValueError(f'invalid data type {data_name}')
    
    gpu_id = 0
    timestep_respacing = '1000'
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = gpu_id
    configs.diffusion.params.timestep_respacing = timestep_respacing
    sampler_dist = DiffusionSampler(configs)
    return sampler_dist

def metrics_ssim(img1, img2, img_idx, shape, column_name):
    if shape == 'BZHW':
        all_df = []
        for b in range(img1.shape[0]):
            all_score = []
            for z in range(img1[b].shape[0]):
                _img1 = img1[b, z]
                _img2 = img2[b, z]
                data_range = max(np.max(_img1), np.max(_img2)) - min(np.min(_img1), np.min(_img2))
                (score, diff) = structural_similarity(_img1, _img2, full=True, data_range=data_range)
                all_score.append(score)
            df = pd.DataFrame(all_score, columns=[column_name])
            df['img'] = img_idx[b]
            all_df.append(df)
        all_df = pd.concat(all_df, axis=0)
    elif shape == 'BHW':
        all_score = []
        for b in range(img1.shape[0]):
            _img1 = img1[b]
            _img2 = img2[b]
            data_range = max(np.max(_img1), np.max(_img2)) - min(np.min(_img1), np.min(_img2))
            (score, diff) = structural_similarity(_img1, _img2, full=True, data_range=data_range)
            all_score.append(score)
        all_df = pd.DataFrame(all_score, columns=[column_name])
        all_df['img'] = img_idx
    return all_df

def max_proj(vars, X_shape):
    if X_shape == 'BZHW':
        max_proj_vars = [np.max(X, axis=1) for X in vars]
        X_shape = 'BHW'
    return max_proj_vars, X_shape

def split_train_test(data_path, data_type, data_split, data):
    if data_type == 'care':
        # split_train_test_path = os.path.join(os.path.dirname(data_path), 'diffface_split_train_test_idx.pkl')
        split_train_test_path = os.path.join(os.path.dirname(data_path), 'Train2TrainValTestv2.pkl')
    elif data_type == 'niddl':
        split_train_test_path = os.path.join(data_path, 'diffface_split_train_test_idx.pkl')
    
    if os.path.exists(split_train_test_path):
        with open(split_train_test_path, 'rb') as f:
            train_test_idx = pickle.load(f)
        if data_type == 'care':
            data = data[train_test_idx[data_split]]
        elif data_type == 'niddl':
            data = [data[i] for i in train_test_idx[data_split]]
        print(f'{data_split} split used')
    return data

def load_data(data_name, data_type, data_split):
    if data_name == 'planaria_2D':
        data_path = '/storage/coda1/p-hl94/0/schaudhary9/siva_projects/CARE-dataset/Denoising_Planaria/train_data/data_label.npz'    
        data = np.load(data_path)
        X = split_train_test(data_path, data_type, data_split, data['X'])
        Y = split_train_test(data_path, data_type, data_split, data['Y'])
        X = np.transpose(X, (0, 2, 3, 4, 1))
        Y = np.transpose(Y, (0, 2, 3, 4, 1))
    else:
        raise ValueError(f'invalid data_name {data_name}')

    return X, Y, data_path


def get_parser(input_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="data name")
    parser.add_argument("--data_type", type=str, help="care or niddl data")
    parser.add_argument("--data_split", type=str, help="train/test/val")

    args = parser.parse_args(input_args)
    return args.data_name, args.data_type, args.data_split


if __name__ == '__main__':
    data_name, data_type, data_split = get_parser(sys.argv[1:])
    
    run = 8
    data = np.load('/storage/coda1/p-hl94/0/schaudhary9/siva_projects/CARE-dataset/Denoising_Planaria/train_data/small_data_label.npz')
    subset = [1, 5, 9]
    X = data['X'][subset, 0, ...]
    Y = data['Y'][subset, 0, ...]
    n2v_pred = data['n2v_pred'][subset]
    care_pred = data['care_pred'][subset]
    X_shape = 'BZHW'


    diff_model = create_diffusion_sampler(data_name)
    
    cfg = {
        'tN_grid': [40, 60, 80],
        'n_samples_grid': [1, 5],
        'repaint_setting': [
            {'repeat_timestep': 10, 'num_repeats': 3, 'mixing': False, 'mixing_stop':50},
            {'repeat_timestep': 20, 'num_repeats': 4, 'mixing': False, 'mixing_stop':50},
            {'repeat_timestep': 30, 'num_repeats': 5, 'mixing': False, 'mixing_stop':50},
            {'repeat_timestep': 10, 'num_repeats': 3, 'mixing': True, 'mixing_stop':50},
            {'repeat_timestep': 20, 'num_repeats': 4, 'mixing': True, 'mixing_stop':50},
            {'repeat_timestep': 30, 'num_repeats': 5, 'mixing': True, 'mixing_stop':50},
        ],
    }
    hparam_grid = list(
        itertools.product(*cfg.values())
    )
    total_runs = len(hparam_grid)
    print(f'total_runs - {total_runs}')
    all_param_df = []
    for i, cfg_value in enumerate(hparam_grid):
        print(cfg_value)
        tN = cfg_value[0]
        n_samples = cfg_value[1]
        repeat_timestep = cfg_value[2]['repeat_timestep']
        num_repeats = cfg_value[2]['num_repeats']
        mixing = cfg_value[2]['mixing']
        mixing_stop = cfg_value[2]['mixing_stop']

        sampler_fn = partial(
            diff_model.repaint_style_sample,
            tN=tN,
            repeat_timestep=repeat_timestep,
            num_repeats=num_repeats,
            mixing=mixing,
            mixing_stop=mixing_stop
        )
        fn_diffusion_denoise = partial(
            diff_model.diffusion_denoise,
            X_shape=X_shape,
            sampler_fn=sampler_fn,
            n_samples=n_samples,
        )
        
        noise_repaint = fn_diffusion_denoise([X])
        n2v_repaint = fn_diffusion_denoise([n2v_pred])
        care_repaint = fn_diffusion_denoise([care_pred])
        img_idx = list(range(X.shape[0]))
        
        [proj_X, proj_Y, proj_n2v_pred, proj_care_pred, proj_noise_repaint, 
         proj_n2v_repaint, proj_care_repaint], proj_X_shape = max_proj(
            [X, Y, n2v_pred, care_pred, noise_repaint, n2v_repaint, care_repaint],
            X_shape
        )
        
        noise_ssim = metrics_ssim(proj_X, proj_Y, img_idx, proj_X_shape, 'noise_ssim')
        n2v_ssim = metrics_ssim(proj_n2v_pred, proj_Y, img_idx, proj_X_shape, 'n2v_ssim') 
        care_ssim = metrics_ssim(proj_care_pred, proj_Y, img_idx, proj_X_shape, 'care_ssim')
        noise_repaint_ssim = metrics_ssim(proj_noise_repaint, proj_Y, img_idx, proj_X_shape, 'noise_repaint_ssim')
        n2v_repaint_ssim = metrics_ssim(proj_n2v_repaint, proj_Y, img_idx, proj_X_shape, 'n2v_repaint_ssim')
        care_repaint_ssim = metrics_ssim(proj_care_repaint, proj_Y, img_idx, proj_X_shape, 'care_repaint_ssim')

        df = [df.set_index('img') for df in [noise_ssim, n2v_ssim, care_ssim, noise_repaint_ssim, n2v_repaint_ssim, care_repaint_ssim]]
        df = pd.concat(df, axis=1)
        df['tN'] = tN
        df['n_samples'] = n_samples
        df['repeat_timestep'] = repeat_timestep
        df['num_repeats'] = num_repeats
        all_param_df.append(df)
    
    all_param_df = pd.concat(all_param_df, axis=0)
    all_param_df.to_pickle(f'/storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/DifFace/results/accuracy_hparam_repaint_planaria_run_{run}.pkl')



    




