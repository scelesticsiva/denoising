import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob
import time
import math
import pickle as pkl
import torchvision as thv

from math import ceil
from omegaconf import OmegaConf
from sampler import BaseSampler
from utils import util_common
from utils import util_image
from torch.utils import data

from inference.data import npz_dataset
from inference.val_utils import filter_val_files

class DiffusionSampler(BaseSampler):
    def sample_func(self, mixing_img_batch, noise=None, start_timesteps=None, bs=4, num_images=1000):
        print('Begining sampling:')
        device = mixing_img_batch.device

        # Mixing time steps in reverse direction
        start_time = 300
        end_time = 50

        # print(np.arange(end_time, start_time, 1)[::-1], device)
        time_steps_to_mix = torch.from_numpy(np.arange(end_time, start_time, 1)[::-1].copy()).to(device)
        const_weights = torch.from_numpy(np.array([0.5] * len(time_steps_to_mix))).to(device)
        # weights = torch.from_numpy(np.linspace(0.8, 0.01, len(time_steps_to_mix))).to(device)
        mixing_weights = dict(zip(time_steps_to_mix, const_weights))

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
                sample = self.diffusion.p_sample_loop_w_mixing(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        mixing_img=mixing_img_batch,
                        time_steps_to_mix=time_steps_to_mix,
                        mixing_weights=mixing_weights,
                        )
            sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)
        return sample

def main(cfg):
    transform = thv.transforms.Normalize(mean=(0.5), std=(0.5))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestep_respacing = '1000'

    configs = OmegaConf.load(cfg.cfg_path)
    configs.gpu_id = cfg.gpu_id
    configs.diffusion.params.timestep_respacing = timestep_respacing

    sampler_dist = DiffusionSampler(configs)

    tN = torch.tensor(cfg.tN).to(device)

    all_data_to_infer = np.load(cfg.noisy_img_npz)[cfg.key_in_npz]
    if cfg.split_fpth and cfg.split_to_infer:
        with open(config.split_fpth, 'rb') as pkl_f:
            splits = pkl.load(pkl_f)
        all_data_to_infer = all_data_to_infer[splits[cfg.split_to_infer]].squeeze(axis=1)
        print(f'Loaded {cfg.split_to_infer} split')

    print(f'Number of datapoints to infer: {len(all_data_to_infer)}')

    all_data_clean = np.load(cfg.clean_img_npz)[cfg.key_in_npz_clean]
    print(f'Number of datapoints (clean): {len(all_data_clean)}')

    assert (len(all_data_to_infer) == len(all_data_clean))
    
    print(f'Device availability:{device}')

    n_batches = int(math.ceil(len(all_data_to_infer) / cfg.bsz))
    master_pred = []

    st = time.time()
    for b_idx in range(n_batches):
        from_ = (b_idx * cfg.bsz)
        to_ = ((b_idx + 1) * cfg.bsz)

        # noisy images to infer
        curr_batch = all_data_to_infer[from_: to_, ...]
        B, C, H, W = curr_batch.shape
        curr_batch = np.reshape(curr_batch, (B * C, 1, H, W))  # Bx1xHxW

        # clean images for mixing
        curr_batch_clean = all_data_clean[from_: to_, ...]
        B, C, H, W = curr_batch_clean.shape
        curr_batch_clean = np.reshape(curr_batch_clean, (B * C, 1, H, W))  # Bx1xHxW

        st = time.time()
        y0 = transform(torch.from_numpy(curr_batch).to(device))
        x0_clean = transform(torch.from_numpy(curr_batch_clean).to(device))
        xN_given_y0 = sampler_dist.diffusion.q_sample(y0, tN)
        x0_sampled = sampler_dist.sample_func(
            mixing_img_batch=x0_clean, # mix with clean images (n2v predictions)
            noise=xN_given_y0,
            start_timesteps=tN,
            bs=len(y0),
            num_images=len(y0),
        )

        # Save the denoised
        curr_pred = x0_sampled.cpu().numpy()
        master_pred.append(curr_pred.reshape(B, C, H, W))

        if (b_idx+1) % 20 == 0:
            print(f'{b_idx + 1} batches done')
    
    to_save = np.concatenate(master_pred, axis=0)
    save_fpth = cfg.save_fpth.replace('.npz', f'_t{cfg.tN}.npz')
    np.savez_compressed(save_fpth, pred=to_save)
    print(f'Saved pred to {save_fpth}, shape of pred: {to_save.shape}, time: {time.time() - st:.4f} seconds')

if __name__ == '__main__':
    from munch import Munch

    config = Munch.fromDict(dict(
        cfg_path='/home/sivark/repos/denoising/open_source/DifFace/configs/sample/iddpm_planaria_LargeMdl.yaml',
        gpu_id='0',
        tN=100, 
        noisy_img_npz='/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/train_data/data_label.npz',
        clean_img_npz='/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/n2v_pred_Train2TrainValTest_val_idxv2.npz',
        save_fpth='/home/joy/project_repos/denoising/data/planaria/Denoising_Planaria/Train2TrainValTest_pred/Diff_mixing_clean_pred_Train2TrainValTest_val_idx.npz',
        key_in_npz='X',
        key_in_npz_clean='pred',
        split_fpth='/home/sivark/supporting_files/denoising/trained_models/planaria_Train2TrainValTestv2.pkl',  # Only required when predicting starting from raw data
        split_to_infer='val_idx',  # Only required when predicting starting from raw data
        bsz=8,
    ))

    main(config)