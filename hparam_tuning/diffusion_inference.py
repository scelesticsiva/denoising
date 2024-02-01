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
            mixing_start_time=100,
            mixing_end_time=50,
            mixing_start_weight=0.8,
            mixing_end_weight=0.01,
            mixing_time_step=1,
            noise=None, 
            start_timesteps=None, 
            bs=4, 
            num_images=1000):
        #print('Begining sampling:')
        if mixing_img_batch is not None:
            device = mixing_img_batch.device

            # Mixing time steps in reverse direction
            start_time = mixing_start_time
            end_time = mixing_end_time
            step_time = mixing_time_step

            time_steps_to_mix = list(range(end_time, start_time, step_time))[::-1]
            weights = torch.from_numpy(np.linspace(mixing_start_weight, mixing_end_weight, len(time_steps_to_mix))).to(device)
            mixing_weights = dict(zip(time_steps_to_mix, weights))

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

    if cfg.do_mixing:
        all_data_to_mix = np.load(cfg.noisy_img_npz)[cfg.mixing_key_in_npz]
        assert (len(all_data_to_infer) == len(all_data_to_mix))

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
        y0 = transform(torch.from_numpy(curr_batch).to(device))

        if cfg.do_mixing:
            curr_batch_to_mix = all_data_to_mix[from_: to_, ...]
            B, C, H, W = curr_batch_to_mix.shape
            curr_batch_to_mix = np.reshape(curr_batch_to_mix, (B * C, 1, H, W))  # Bx1xHxW
            mixing_img_batch = transform(torch.from_numpy(curr_batch_to_mix).to(device))

        st = time.time()

        master_curr_pred = None
        for _ in range(cfg.n_samples):
            xN_given_y0 = sampler_dist.diffusion.q_sample(y0, tN)
            if cfg.do_mixing:
                x0_sampled = sampler_dist.sample_func(
                    mixing_img_batch=mixing_img_batch,
                    mixing_start_time=cfg.mixing_start_time,
                    mixing_end_time=cfg.mixing_end_time,
                    mixing_start_weight=cfg.mixing_start_weight,
                    mixing_end_weight=cfg.mixing_end_weight,
                    mixing_time_step=cfg.mixing_time_step,
                    noise=xN_given_y0,
                    start_timesteps=tN,
                    bs=len(y0),
                    num_images=len(y0),
                )
            else:
                x0_sampled = sampler_dist.sample_func(
                    noise=xN_given_y0,
                    start_timesteps=tN,
                    bs=len(y0),
                    num_images=len(y0),
                )
            if master_curr_pred is None:
                master_curr_pred = x0_sampled
            else:
                master_curr_pred += x0_sampled
        master_curr_pred = master_curr_pred / torch.tensor(cfg.n_samples)
        curr_pred = master_curr_pred.cpu().numpy()
        master_pred.append(curr_pred.reshape(B, C, H, W))

        #if (b_idx+1) % 20 == 0:
        #    print(f'{b_idx + 1} batches done')

    to_save = np.concatenate(master_pred, axis=0)
    #save_fpth = cfg.save_fpth.replace('.npz', f'_t{cfg.tN}.npz')
    save_fpth = cfg.save_fpth
    np.savez_compressed(save_fpth, pred=to_save)
    print(f'Saved pred to {save_fpth}, shape of pred: {to_save.shape}, time: {time.time() - st:.4f} seconds')

if __name__ == '__main__':
    from munch import Munch

    # an npz file containing all the tensors for this split: 
    #   clean:    the clean data
    #   noisy:    the noisy data
    #   n2v_only: the outputs of N2V (w/o diffusion)
    #data_npz = '/home/mds/data/denoising/datasets/clean_noisy_n2v_val_set.npz'
    data_npz = '/home/mds/data/denoising/datasets/clean_noisy_n2v_val_set_subset10.npz'

    # output dir to save all the runs
    #tune_dir = '/home/mds/data/denoising/hparam_tuning/diffusion_outputs/val_all/v1'
    tune_dir = '/home/mds/data/denoising/hparam_tuning/diffusion_outputs/val_subset10/mixing_test'
    os.makedirs(tune_dir, exist_ok=True)

    # base config params
    diff_mdl_cfg = '/home/sivark/repos/denoising/open_source/DifFace/configs/sample/iddpm_planaria_LargeMdl_80percent_train.yaml'

    config = dict(
        cfg_path=diff_mdl_cfg,
        noisy_img_npz=data_npz,
        key_in_npz='noisy',
        gpu_id='0',
        split_fpth=None,
        split_to_infer=None,
        bsz=32,
    )
    
    # hparams (grid search)
    # each sub-array will be treated independently.
    # params that depend on each other (e.g. mixing start+end) should be grouped together.
    # TODO load this from a file

    # NOTE 'mixing_key_in_npz' controls which data to mix in:
    #    noisy:    mixes in the raw data
    #    n2v_only: mixes in the 'pseudo-clean' data from running N2V

    hparam_grid = [
        [
            {'tN': 100},
        ],
        [
            {'n_samples': 1},
            #{'n_samples': 5},
            #{'n_samples': 10},
            #{'n_samples': 20},
        ],
        [
            #{'do_mixing': False},
            {'do_mixing': True},
        ],
        [
            #{'mixing_key_in_npz': 'noisy'},
            {'mixing_key_in_npz': 'n2v_only'},
        ],
        [
            {'mixing_start_time': 100, 'mixing_end_time': 20, 'mixing_time_step': 1},
            #{'mixing_start_time': 100, 'mixing_end_time': 50, 'mixing_time_step': 1},
        ],
        [
            #{'mixing_start_weight': 0.8, 'mixing_end_weight': 0.01},
            {'mixing_start_weight': 0.3, 'mixing_end_weight': 0.01},
        ],
    ]

    hparam_grid = list(
        itertools.product(*hparam_grid)
    )
    total_runs = len(hparam_grid)
    print(f"Total hparam grid size: {total_runs}")
    for i, hparams in enumerate(hparam_grid):
        hparam_dict = {k: v for d in hparams for k, v in d.items()}
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
