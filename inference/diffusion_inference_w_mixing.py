import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob
import time
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
        time_steps_to_mix = list(range(end_time, start_time, 1))[::-1]
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

    all_val_files = filter_val_files(cfg.noisy_img_dir, val_files=cfg.val_files, ext='*.npz')
    dataset = npz_dataset.NPZ3DDataset(all_val_files, key='pred')
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(f'Device availability:{device}')

    for batch in dataloader:
        st = time.time()
        y0 = transform(torch.permute(batch['img'].to(device), (1, 0, 2, 3)))
        xN_given_y0 = sampler_dist.diffusion.q_sample(y0, tN)
        x0_sampled = sampler_dist.sample_func(
            mixing_img_batch=y0,
            noise=xN_given_y0,
            start_timesteps=tN,
            bs=len(y0),
            num_images=len(y0),
        )

        # Save the denoised
        assert len(batch['fpth']) == 1
        save_fpth = os.path.join(cfg.save_dir, os.path.basename(batch['fpth'][0]))
        to_save = np.transpose(x0_sampled.cpu().numpy(), (1, 0, 2, 3)).squeeze(axis=0)
        np.savez_compressed(save_fpth, pred=to_save)
        print(f'Saved pred to {save_fpth}, shape of pred: {to_save.shape}, time: {time.time() - st:.4f} seconds')

if __name__ == '__main__':
    from munch import Munch

    config = Munch.fromDict(dict(
        cfg_path='/home/sivark/repos/denoising/open_source/DifFace/configs/sample/iddpm_planaria_LargeMdl.yaml',
        gpu_id='0',
        tN=300,
        noisy_img_dir='/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/n2v_pred/condition_1/',
        save_dir='/home/sivark/supporting_files/denoising/data/planaria/Denoising_Planaria/diffusion_pred/t500_from_n2v_mixing/condition_1/',
        val_files=['EXP280_Smed_live_RedDot1_slide_mnt_N3_stk1',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0003',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0005',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0013',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0009'],
    ))

    main(config)



