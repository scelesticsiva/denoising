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


class DiffusionSampler(BaseSampler):
    def sample_func(self, noise=None, start_timesteps=None, bs=4, num_images=1000, save_dir=None):
        print('Begining sampling:')

        h = w = self.configs.im_size
        if self.num_gpus != 0:
            total_iters = ceil(num_images / (bs * self.num_gpus))
        else:
            total_iters = ceil(num_images / (bs * 1))
        for ii in range(total_iters):
            if self.rank == 0 and self.display:
                print(f'Processing: {ii+1}/{total_iters}')
            if noise == None:
                if self.num_gpus != 0:
                    noise = torch.randn((bs, 1, h, w), dtype=torch.float32).cuda()
                else:
                    noise = torch.randn((bs, 1, h, w), dtype=torch.float32)
            if 'ddim' in self.configs.diffusion.params.timestep_respacing:
                sample = self.diffusion.ddim_sample_loop(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        eta=0.0,
                        )
            else:
                sample = self.diffusion.p_sample_loop(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        )
            sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)
        return sample
    
    def repaint_style_sample(self, noise=None, tN=100, bs=1, repeat_timestep=1, num_repeats=1, mixing=False, mixing_stop=50):
        print('Begining sampling:')
        
        h = w = self.configs.im_size
        shape = (bs, 1, h, w)
        if noise != None:
            img = noise
        else:
            img = torch.randn((bs, 1, h, w))

        while tN > 0:
            indices = list(range(tN)[::-1])[0:repeat_timestep]
            print(f'repainting between {indices[0]}-{indices[-1]} for {num_repeats} times')
            for rp in range(num_repeats):
                # img = self.diffusion.q_sample(img, indices[0])
                for i in indices:
                    t = torch.tensor([i] * shape[0], device=img.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            img,
                            t,
                            clip_denoised=True,
                            denoised_fn=None,
                            model_kwargs=None,
                        )
                        img = out["sample"]
                    if mixing == True and i > mixing_stop:
                        xN_given_y0 = self.diffusion.q_sample(noise, torch.tensor(i))
                        img = 0.5*img + 0.5*xN_given_y0
            tN -= repeat_timestep
        sample = util_image.normalize_th(img, reverse=True).clamp(0.0, 1.0)
        return sample

def create_diffusion_sampler(data_name):
    if data_name == 'planaria_2D':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_care_planaria_2D.yaml'
    elif data_name == 'tribolium_2D':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_care_tribolium_2D.yaml'
    elif data_name == 'flywing':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_care_flywing.yaml'
    elif data_name == 'synthetic_tubulin_gfp':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_care_synthetic_tubulin_gfp.yaml'
    elif data_name == 'synthetic_tubulin_granules_channel_granules':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_care_synthetic_tubulin_granules_channel_granules.yaml'
    elif data_name == 'synthetic_tubulin_granules_channel_tubules':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_care_synthetic_tubulin_granules_channel_tubules.yaml'
    elif data_name == 'niddl':
        cfg_path = '/Users/schaudhary/siva_projects/DifFace/configs/sample/iddpm_niddl_wb.yaml'
    else:
        raise ValueError(f'invalid data type {data_name}')
    
    gpu_id = 0
    timestep_respacing = '1000'
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = gpu_id
    configs.diffusion.params.timestep_respacing = timestep_respacing
    sampler_dist = DiffusionSampler(configs)
    return sampler_dist

def diffusion_denoise_repaint(X, sampler_dist, X_shape, tN, n_samples, repeat_timestep, num_repeats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = thv.transforms.Normalize(mean=(0.5), std=(0.5))
    tN = torch.tensor(tN, device=device)
    if X_shape == 'BZHWC':
        B, Z, H, W, C = X.shape
        X = np.reshape(X, [-1, H, W, C])
        X = X[:, :, :, 0]
    elif X_shape == 'BZHW':
        B, Z, H, W = X.shape
        X = np.reshape(X, [-1, H, W])
    elif X_shape == 'BHWC':
        B, H, W, C = X.shape
        X = X[:, :, :, 0]
    elif X_shape == 'BHW':
        X = X
    else:
        raise ValueError(f'invalid X_shape {X_shape}')
    bs = X.shape[0]
    y0 = torch.tensor(X, dtype=torch.float32, device=device)
    y0 = y0[:, None, :, :]
    y0 = transform(y0)
    all_samples = []
    for n in range(n_samples):
        xN_given_y0 = sampler_dist.diffusion.q_sample(y0, tN)
        x0_sampled = sampler_dist.repaint_style_sample(xN_given_y0, tN, bs, repeat_timestep, num_repeats)
        all_samples.append(x0_sampled)
    x0_sampled = torch.mean(torch.stack(all_samples, dim=0), dim=0)
    if X_shape == 'BZHWC':
        x0_sampled = np.reshape(x0_sampled.cpu().numpy(), [B, Z, H, W])
    elif X_shape == 'BZHW':
        x0_sampled = np.reshape(x0_sampled.cpu().numpy(), [B, Z, H, W])
    elif X_shape == 'BHWC':
        x0_sampled = x0_sampled.cpu().numpy()[:, 0, :, :]
    elif X_shape == 'BHW':
        x0_sampled = x0_sampled.cpu().numpy()[:, 0, :, :]
    return x0_sampled


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

def get_parser(input_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="data name")
    parser.add_argument("--data_type", type=str, help="care or niddl data")
    args = parser.parse_args(input_args)
    return args.data_name, args.data_type


if __name__ == '__main__':
    data_name, data_type = get_parser(sys.argv[1:])

    data = np.load('/Users/schaudhary/siva_projects/Denoising_Planaria/train_data/small_data_label.npz')
    subset = [1, 5]
    X = data['X'][subset, 0, ...]
    Y = data['Y'][subset, 0, ...]
    n2v_pred = data['n2v_pred'][subset]
    care_pred = data['care_pred'][subset]
    X_shape = 'BZHW'

    diff_model = create_diffusion_sampler(data_name)
    
    tN_grid = [4]
    #n_samples_grid = [1, 5, 10, 15, 20]
    n_samples_grid = [1]
    # TODO mixing has a bug - leave it turned off for now
    # NOTE 'mixing_key_in_npz' controls which data to mix in:
    #    noisy:    mixes in the raw data
    #    n2v_only: mixes in the 'pseudo-clean' data from running N2V
    repaint_setting = [
        {'repeat_timestep': 2, 'num_repeats': 2, 'mixing': False, 'mixing_stop': 50},
        {'repeat_timestep': 2, 'num_repeats': 2, 'mixing': False, 'mixing_stop': 50},
        {'repeat_timestep': 1, 'num_repeats': 2, 'mixing': False, 'mixing_stop': 50},
    ]
    hparam_grid = list(
        itertools.product(tN_grid, n_samples_grid, repaint_setting)
    )
    total_runs = len(hparam_grid)
    print(f'total_runs - {total_runs}')
    all_param_df = []
    for i, (tN, n_samples, repaint_setting) in enumerate(hparam_grid):
        print(hparam_grid[i])

        tN = tN
        n_samples = n_samples
        repeat_timestep = repaint_setting['repeat_timestep']
        num_repeats = repaint_setting['num_repeats']
        mixing = repaint_setting['mixing']
        mixing_stop = repaint_setting['mixing_stop']

        fn_diffusion_denoise = partial(
            diffusion_denoise_repaint,
            sampler_dist=diff_model,
            X_shape=X_shape,
            tN=tN, 
            n_samples=n_samples,
            repeat_timestep=repeat_timestep,
            num_repeats=num_repeats,
            mixing=mixing,
            mixing_stop=mixing_stop
        )
        
        noise_repaint = fn_diffusion_denoise(X)
        n2v_repaint = fn_diffusion_denoise(n2v_pred)
        care_repaint = fn_diffusion_denoise(care_pred)
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

        df = [df.set_index('img') for df in [noise_ssim, n2v_ssim, noise_repaint_ssim, n2v_repaint_ssim]]
        df = pd.concat(df, axis=1)
        df['tN'] = tN
        df['n_samples'] = n_samples
        df['repeat_timestep'] = repeat_timestep
        df['num_repeats'] = num_repeats
        all_param_df.append(df)
    
    all_param_df = pd.concat(all_param_df, axis=0)
    all_param_df.to_pickle('/Users/schaudhary/siva_projects/DifFace/accuracy_hparam_repaint_planaria.pkl')



    




