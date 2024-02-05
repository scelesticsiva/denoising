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

def get_parser(input_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="data name")
    parser.add_argument("--data_type", type=str, help="care or niddl data")
    parser.add_argument("--data_path", type=str, help="path of npz file")
    parser.add_argument("--data_key", type=str, help="key in npz file")
    parser.add_argument("--data_shape", type=str, help="BZHW/BHW/BZHWC")
    parser.add_argument("--bs", type=str, help="batch size")
    parser.add_argument("--save_path", type=str, help="path to save diff pred npz files")

    args = parser.parse_args(input_args)
    return args.data_name, args.data_type, args.data_path, args.data_key, args.data_shape, args.bs, args.save_path


if __name__ == '__main__':
    
    data_name, data_type, data_path, data_key, data_shape, bs, save_path = get_parser(sys.argv[1:])
    
    X = np.load(data_path)[data_key]
    npz_name = os.path.basename(X)[::-4]

    diff_model = create_diffusion_sampler(data_name)
    
    cfg = {
        'tN_grid': [60],
        'n_samples_grid': [1, 5],
        'repaint_setting': [
            {'repeat_timestep': 20, 'num_repeats': 4, 'mixing': False, 'mixing_stop':50},
        ],
    }
    hparam_grid = list(
        itertools.product(*cfg.values())
    )
    total_runs = len(hparam_grid)
    print(f'total_runs - {total_runs}')
    all_param_df = []
    for i, cfg_value in enumerate(hparam_grid):        
        print(hparam_grid[i])
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
            X_shape=data_shape,
            sampler_fn=sampler_fn,
            n_samples=n_samples,
        )

        num_batches = X.shape[0]
        if num_batches%bs == 0:
            num_iters = num_batches//bs
        else:
            num_iters = num_batches//bs + 1
        all_diff_pred = []
        for it in range(num_iters):
            curr_batch = X[it*bs:min((it+1)*bs, num_batches)]
            diff_pred = fn_diffusion_denoise([curr_batch])
            all_diff_pred.append(diff_pred)
        all_diff_pred = np.concatenate(all_diff_pred, axis=0)

        pred_save_path = os.path.join(save_path, f'{npz_name}_repaint.npz')
        np.savez(pred_save_path, data_key=data_key)
        cfg_save_path = os.path.join(save_path, f'{npz_name}_repaint_cfg.npz')
        curr_cfg = dict(zip(cfg.keys(), cfg_value))
        with open(cfg_save_path, 'wb') as f:
            pickle.dump(curr_cfg, f)



    




