import json
import numpy as np
import os
import torch
import glob
import time
import math
import pickle as pkl
import torchvision as thv
import itertools

from math import ceil
from omegaconf import OmegaConf
from sampler import BaseSampler
from utils import util_common
from utils import util_image
from torch.utils import data

from inference.data import npz_dataset
from inference.val_utils import filter_val_files

class DiffusionSampler(BaseSampler):
    def sample_func(
            self, 
            mixing_img_batch=None, 
            mixing_start_time=300,
            mixing_end_time=50,
            mixing_time_step=1,
            noise=None, 
            start_timesteps=None, 
            bs=4, 
            num_images=1000):
        #print('Begining sampling:')
        if mixing_img_batch is not None:
            print("doing mixing")
            device = mixing_img_batch.device

            # Mixing time steps in reverse direction
            start_time = mixing_start_time
            end_time = mixing_end_time
            step_time = mixing_time_step

            # print(np.arange(end_time, start_time, 1)[::-1], device)
            time_steps_to_mix = torch.from_numpy(np.arange(end_time, start_time, step_time)[::-1].copy()).to(device)
            const_weights = torch.from_numpy(np.array([0.5] * len(time_steps_to_mix))).to(device)
            # weights = torch.from_numpy(np.linspace(0.8, 0.01, len(time_steps_to_mix))).to(device)
            mixing_weights = dict(zip(time_steps_to_mix, const_weights))

        h = w = self.configs.im_size
        if self.num_gpus != 0:
            total_iters = ceil(num_images / (bs * self.num_gpus))
        else:
            total_iters = ceil(num_images / (bs * 1))
        for ii in range(total_iters):
            #if self.rank == 0 and self.display:
            #    print(f'Processing: {ii+1}/{total_iters}')
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
            elif mixing_img_batch is not None:
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

    #print(f'Number of datapoints to infer: {len(all_data_to_infer)}')
    #print(f'Device availability:{device}')

    n_batches = int(math.ceil(len(all_data_to_infer) / cfg.bsz))
    master_pred = []

    st = time.time()
    for b_idx in range(n_batches):
        from_ = (b_idx * cfg.bsz)
        to_ = ((b_idx + 1) * cfg.bsz)
        curr_batch = all_data_to_infer[from_: to_, ...]
        B, C, H, W = curr_batch.shape
        curr_batch = np.reshape(curr_batch, (B * C, 1, H, W))  # Bx1xHxW

        st = time.time()
        y0 = transform(torch.from_numpy(curr_batch).to(device))
        xN_given_y0 = sampler_dist.diffusion.q_sample(y0, tN)

        if cfg.do_mixing:
            mixing_img_batch = y0
            mixing_start_time = cfg.mixing_start_time
            mixing_end_time = cfg.mixing_end_time
            mixing_time_step = cfg.mixing_time_step
        else:
            mixing_img_batch = None
            mixing_start_time = None
            mixing_end_time = None
            mixing_time_step = None

        x0_sampled = sampler_dist.sample_func(
            mixing_img_batch=mixing_img_batch,
            mixing_start_time=mixing_start_time,
            mixing_end_time=mixing_end_time,
            noise=xN_given_y0,
            start_timesteps=tN,
            bs=len(y0),
            num_images=len(y0),
        )

        # Save the denoised
        curr_pred = x0_sampled.cpu().numpy()
        master_pred.append(curr_pred.reshape(B, C, H, W))

        #if (b_idx+1) % 20 == 0:
        #    print(f'{b_idx + 1} batches done')

    to_save = np.concatenate(master_pred, axis=0)
    #save_fpth = cfg.save_fpth.replace('.npz', f'_t{cfg.tN}.npz')
    save_fpth = cfg.save_fpth
    np.savez_compressed(save_fpth, pred=to_save)
    #print(f'Saved pred to {save_fpth}, shape of pred: {to_save.shape}, time: {time.time() - st:.4f} seconds')

if __name__ == '__main__':
    from munch import Munch

    # an npz file containing all the tensors for this split: 
    #   clean:    the clean data
    #   noisy:    the noisy data
    #   n2v_only: the outputs of N2V (w/o diffusion)
    data_npz = '/home/mds/data/denoising/datasets/clean_noisy_n2v_val_set_subset10.npz',

    # output dir to save all the runs
    tune_dir = '/home/mds/data/denoising/hparam_tuning/diffusion_outputs/val_subset10/v2'
    os.makedirs(tune_dir, exist_ok=True)

    # base config params
    diff_mdl_cfg = '/home/sivark/repos/denoising/open_source/DifFace/configs/sample/iddpm_planaria_LargeMdl.yaml',
    config = dict(
        cfg_path=diff_mdl_cfg,
        noisy_img_npz=data_npz,
        key_in_npz='n2v_only',
        gpu_id='0',
        split_fpth=None,
        split_to_infer=None,
        bsz=32,
    )
    
    # hparams (grid search)
    # TODO: increase search space
    tN_grid = [100, 300]
    mixing_setting_grid = [
        {'do_mixing': False, 'mixing_start_time': None, 'mixing_end_time': None, 'mixing_time_step': None,},
        {'do_mixing': True, 'mixing_start_time': 300, 'mixing_end_time': 50, 'mixing_time_step': 1},
    ]
    hparam_grid = list(
        itertools.product(tN_grid, mixing_setting_grid)
    )
    total_runs = len(hparam_grid)
    print(f"Total hparam grid size: {total_runs}")
    for i, (tN, mixing_setting) in enumerate(hparam_grid):
        hparam_dict = {'tN': tN}
        hparam_dict.update(mixing_setting)
        print(f"Run {i}/{total_runs}")
        print(hparam_dict)
        run_path = os.path.join(tune_dir, f'run{i}')
        save_fpth = f"{run_path}.npz"
        hparam_dict["save_fpth"] = save_fpth
        # save hparams
        with open(f"{run_path}.json", "w") as f:
            json.dump(hparam_dict, f)
        # update config + run
        config.update(hparam_dict)
        st = time.time()
        main(Munch.fromDict(config))
        print(f'Time: {time.time() - st:.4f} seconds')
